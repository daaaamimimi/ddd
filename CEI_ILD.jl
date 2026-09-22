module CEI_ILD

using LinearAlgebra
using Printf

export cei_ild, cei_ild_from_il, cei_ild_sparams, cei_ild_s4p,
       read_s4p, sdd21_from_sparams, engineering_settings,
       print_ild_summary, write_ild_csv

# 算法出处：OIF-CEI-05.2 §12.2.1.1、§14.2.6.4。
# https://www.oiforum.com/wp-content/uploads/OIF-CEI-05.2.pdf
# 注意：§14 是 28G-MR，不是 56G-VSR-NRZ 参数表。
# 56G-VSR-NRZ 的完整草案参数尚未核实；不内置“标准合规”判断。
# 所有频率使用 Hz，时间使用秒，IL 使用正损耗 dB。

"""
    engineering_settings(; fb_hz=56e9)

可运行的工程示例配置，不是已核实的 CEI-56G-VSR-NRZ 标准 profile。
9 ps 仅作为示例边沿时间；必须与实际使用的草案/测试方法核对。
系数边界默认不启用，FOM 限值默认不设置。
"""
function engineering_settings(; fb_hz::Real=56e9)
    fb = Float64(fb_hz)
    return (
        fb_hz=fb,
        ffit_min_hz=50e6,
        ffit_max_hz=fb,
        ffom_min_hz=50e6,
        ffom_max_hz=0.75 * fb,
        tr_s=9e-12,
        tf_s=9e-12,
        fr_hz=0.75 * fb,
        coeff_min=nothing,   # 顺序 (a0, a1, a2, a4)，核实后填写
        coeff_max=nothing,   # 例如传入一个包含四个数值的元组
        fom_limit_db=nothing,
        grid_mode=:strict,
        df_hz=10e6,         # 只用于 :linear_il；严格模式保留原频点
        allow_coarse=false,
        profile_name="56G NRZ engineering example; VSR draft parameters UNVERIFIED",
    )
end

_basis(f, fb) = hcat(ones(length(f)), sqrt.(f ./ fb), f ./ fb, (f ./ fb).^2)

function _positive(x, name)
    isfinite(x) && x > 0 || throw(ArgumentError("$name 必须是有限正数。"))
end

function _bounds(value, fallback, name)
    value === nothing && return fill(fallback, 4)
    length(value) == 4 || throw(ArgumentError("$name 必须按 (a0,a1,a2,a4) 提供四个数。"))
    a = Float64.(collect(value))
    any(isnan, a) && throw(ArgumentError("$name 不能含 NaN。"))
    return a
end

# 在频段内选取原始点，并要求两端点均存在；不做隐式截短或外推。
function _strict_band(f, y, lo, hi)
    tol = max(0.01, 1e-10 * hi)  # Hz；只容许浮点端点误差
    idx = findall((f .>= lo - tol) .& (f .<= hi + tol))
    length(idx) >= 2 || throw(ArgumentError("频段 $lo 到 $hi Hz 内的点数不足。"))
    fx, yy = f[idx], y[idx]
    abs(first(fx) - lo) <= tol && abs(last(fx) - hi) <= tol ||
        throw(ArgumentError("数据必须包含频段两端点；可显式使用 grid_mode=:linear_il 进行 IL(dB) 插值。"))
    d = diff(fx)
    dtol = max(0.01, 1e-6 * d[1])
    maximum(abs.(d .- d[1])) <= dtol ||
        throw(ArgumentError("严格模式要求等间隔频率；请重新导出，或显式选择 :linear_il。"))
    return fx, yy, maximum(d)
end

function _uniform_grid(lo, hi, df)
    n = max(1, ceil(Int, (hi - lo) / df - 1e-10))
    n <= 10_000_000 || throw(ArgumentError("目标频点过多；请增大 df_hz。"))
    return collect(range(lo, hi; length=n + 1))
end

# 仅插值 IL(dB)，不直接插值带有长时延相位旋转的复数 S21。
function _interp_il(f, y, x)
    first(x) >= first(f) && last(x) <= last(f) ||
        throw(ArgumentError("数据未覆盖请求频段；禁止外推。"))
    out = Vector{Float64}(undef, length(x))
    for k in eachindex(x)
        j = searchsortedlast(f, x[k])
        if f[j] == x[k]
            out[k] = y[j]
        else
            j < length(f) || throw(ArgumentError("插值点超出数据范围。"))
            t = (x[k] - f[j]) / (f[j + 1] - f[j])
            out[k] = (1 - t) * y[j] + t * y[j + 1]
        end
    end
    return out
end

function _fit_coefficients(X, il, lower, upper)
    # 拟合残差平方的权重为 |Sdd21|^2。
    # 行乘子 g=|Sdd21|，且在所有固定系数/重拟合迭代中保持不变。
    g = 10.0 .^ (-il ./ 20)
    all(isfinite, g) && maximum(g) > 0 ||
        throw(ArgumentError("插损动态范围导致拟合权重溢出/下溢。"))
    g ./= maximum(g)  # 全局缩放不改变最小二乘解
    a = zeros(4)
    fixed = falses(4)
    order = (4, 2, 3, 1)  # a4 -> a1 -> a2 -> a0
    names = (:a0, :a1, :a2, :a4)
    history = NamedTuple[]
    tol = 1e-10

    # 每次只固定一个系数；最多固定四个后，最后再求解/检查一次。
    for iteration in 0:4
        free = findall(.!fixed)
        if !isempty(free)
            locked = findall(fixed)
            rhs = isempty(locked) ? copy(il) : il - X[:, locked] * a[locked]
            A = X[:, free] .* reshape(g, :, 1)
            rank(A) == length(free) ||
                throw(ArgumentError("加权拟合矩阵秩不足；检查频段、动态范围和端口映射。"))
            a[free] = A \ (rhs .* g)  # 避免显式求逆 (F'F)^(-1)
        end

        # 先处理下限，再处理上限；同类越界按标准顺序逐一固定并重拟合。
        # 上限固定后若使其余自由系数低于下限，也重新检查下限。
        chosen, side, bound = 0, :none, 0.0
        for j in order
            if !fixed[j] && a[j] < lower[j] - tol
                chosen, side, bound = j, :lower, lower[j]
                break
            end
        end
        if chosen == 0
            for j in order
                if !fixed[j] && a[j] > upper[j] + tol
                    chosen, side, bound = j, :upper, upper[j]
                    break
                end
            end
        end
        if chosen == 0
            all(isfinite, a) || error("拟合系数不是有限数。")
            return a, history
        end
        a[chosen] = bound
        fixed[chosen] = true
        push!(history, (iteration=iteration + 1, coefficient=names[chosen],
                        side=side, value=bound))
    end
    error("系数约束未收敛；检查上下限设置。")
end

"""
    cei_ild_from_il(f_hz, il_db; fb_hz, ffit_min_hz, ffit_max_hz,
                   ffom_max_hz, tr_s, ...)

从正损耗 IL(dB) 计算拟合、ILD 和加权 FOM。
参数必须来自实际采用的草案/测试方法；函数不会推断 CEI 子标准参数。

`grid_mode=:strict`：原频点必须等间隔且包含拟合/FOM 两端点。
`grid_mode=:linear_il`：分别建立包含两端点的均匀拟合/FOM 网格，
对 IL(dB) 线性插值。`df_hz` 为目标最大间隔，实际间隔可能略小。
原数据间隔 >10 MHz 时默认报错；插值不会恢复漏测的窄带纹波。

`coeff_min/coeff_max` 顺序均为 (a0,a1,a2,a4)；nothing 表示该侧无约束。
`meets_user_limit` 仅为用户数值限值比较，不表示 CEI 整体合规。
"""
function cei_ild_from_il(f_hz::AbstractVector{<:Real},
                         il_db::AbstractVector{<:Real};
                         fb_hz::Real,
                         ffit_min_hz::Real,
                         ffit_max_hz::Real,
                         ffom_max_hz::Real,
                         tr_s::Real,
                         ffom_min_hz::Real=ffit_min_hz,
                         tf_s::Real=tr_s,
                         fr_hz::Real=0.75 * fb_hz,
                         coeff_min=nothing,
                         coeff_max=nothing,
                         fom_limit_db=nothing,
                         grid_mode::Symbol=:strict,
                         df_hz::Real=10e6,
                         allow_coarse::Bool=false,
                         profile_name::AbstractString="User settings; not a verified compliance profile")
    length(f_hz) == length(il_db) || throw(DimensionMismatch("频率和 IL 长度不同。"))
    length(f_hz) >= 4 || throw(ArgumentError("至少需要四个频点。"))
    f, il = Float64.(collect(f_hz)), Float64.(collect(il_db))
    all(isfinite, f) && all(f .>= 0) && all(diff(f) .> 0) ||
        throw(ArgumentError("频率必须有限、非负、严格递增且不重复，单位 Hz。"))
    for (x, name) in ((fb_hz,"fb_hz"), (ffit_min_hz,"ffit_min_hz"),
                      (ffit_max_hz,"ffit_max_hz"), (ffom_min_hz,"ffom_min_hz"),
                      (ffom_max_hz,"ffom_max_hz"), (tr_s,"tr_s"),
                      (tf_s,"tf_s"), (fr_hz,"fr_hz"), (df_hz,"df_hz"))
        _positive(x, name)
    end
    ffit_min_hz <= ffom_min_hz < ffom_max_hz <= ffit_max_hz ||
        throw(ArgumentError("FOM 频段必须包含于拟合频段。"))
    grid_mode in (:strict, :linear_il) || throw(ArgumentError("不支持该 grid_mode。"))
    if fom_limit_db !== nothing
        isfinite(fom_limit_db) && fom_limit_db >= 0 || throw(ArgumentError("FOM 限值必须有限且非负。"))
    end
    lower, upper = _bounds(coeff_min, -Inf, "coeff_min"), _bounds(coeff_max, Inf, "coeff_max")
    all(lower .<= upper) && all(lower .< Inf) && all(upper .> -Inf) ||
        throw(ArgumentError("系数上下限无效。"))

    if grid_mode == :strict
        ff, yf, maxstep = _strict_band(f, il, ffit_min_hz, ffit_max_hz)
        fm, ym, _ = _strict_band(f, il, ffom_min_hz, ffom_max_hz)
    else
        first(f) <= ffit_min_hz && last(f) >= ffit_max_hz ||
            throw(ArgumentError("数据必须覆盖整个拟合频段；不能用 FOM 频段代替。"))
        left = max(1, searchsortedlast(f, Float64(ffit_min_hz)))
        right = min(length(f), searchsortedfirst(f, Float64(ffit_max_hz)))
        maxstep = maximum(diff(f[left:right]))
        ff = _uniform_grid(Float64(ffit_min_hz), Float64(ffit_max_hz), Float64(df_hz))
        fm = _uniform_grid(Float64(ffom_min_hz), Float64(ffom_max_hz), Float64(df_hz))
        yf, ym = _interp_il(f, il, ff), _interp_il(f, il, fm)
    end
    length(ff) >= 4 || throw(ArgumentError("拟合频段至少需要四个频点。"))
    all(isfinite, yf) && all(isfinite, ym) ||
        throw(ArgumentError("使用频段内含 NaN/Inf 或零传输点；请检查测量噪声底、端口和数据。"))
    coarse = maxstep > 10e6 * (1 + 1e-6) ||
             maximum(diff(ff)) > 10e6 * (1 + 1e-6) ||
             maximum(diff(fm)) > 10e6 * (1 + 1e-6)
    coarse && !allow_coarse && throw(ArgumentError(
        "原始/计算频点间隔超过 10 MHz。请重新测量或仿真；工程预览须显式设 allow_coarse=true。"))
    coarse && @warn "粗频点工程计算：插值不能恢复漏测纹波，不应用于标准判定。"
    min(minimum(yf), minimum(ym)) < -1e-6 && @warn "使用频段存在负损耗；检查符号、端口、参考阻抗或有源增益。"

    X = _basis(ff, fb_hz)
    a, history = _fit_coefficients(X, yf, lower, upper)
    fit_il = X * a
    fit_fom = _basis(fm, fb_hz) * a
    ild, ild_fom = yf - fit_il, ym - fit_fom
    ft_hz = 0.2365 / min(tr_s, tf_s)
    W = sinc.(fm ./ fb_hz).^2 ./ (1 .+ (fm ./ ft_hz).^4) ./ (1 .+ (fm ./ fr_hz).^8)
    q = W .* ild_fom.^2
    N = length(fm)
    fom = sqrt(sum(q) / N)  # 分母 N，不是 sum(W)；不是 (W .* ILD).^2
    limit_status = fom_limit_db === nothing ? missing : fom <= fom_limit_db

    settings = (profile_name=String(profile_name), fb_hz=Float64(fb_hz),
        ffit_min_hz=Float64(ffit_min_hz), ffit_max_hz=Float64(ffit_max_hz),
        ffom_min_hz=Float64(ffom_min_hz), ffom_max_hz=Float64(ffom_max_hz),
        tr_s=Float64(tr_s), tf_s=Float64(tf_s), ft_hz=Float64(ft_hz), fr_hz=Float64(fr_hz),
        coeff_min=Tuple(lower), coeff_max=Tuple(upper),
        fom_limit_db=fom_limit_db, grid_mode=grid_mode)
    return (fom_ild_db=fom,
        ild_rms_unweighted_db=sqrt(sum(abs2, ild_fom) / N),
        coefficients=(a0=a[1], a1=a[2], a2=a[3], a4=a[4]),
        fit_history=history, settings=settings,
        f_fit_hz=ff, il_db=yf, il_fit_db=fit_il, ild_db=ild,
        f_fom_hz=fm, il_fom_db=ym, il_fit_fom_db=fit_fom,
        ild_fom_db=ild_fom, weights=W, weighted_ild_squared=q,
        cumulative_fom_db=sqrt.(cumsum(q) ./ N),
        n_fit=length(ff), n_fom=N,
        original_max_step_hz=maxstep, coarse_sampling=coarse,
        bounds_applied=coeff_min !== nothing || coeff_max !== nothing,
        meets_user_limit=limit_status)
end

"""从复数 Sdd21 或线性幅度计算；不能传入 dB 值。DC 的零传输可保留，DC 不参与拟合。"""
function cei_ild(f_hz::AbstractVector{<:Real}, sdd21::AbstractVector{<:Number}; kwargs...)
    length(f_hz) == length(sdd21) || throw(DimensionMismatch("频率和 Sdd21 长度不同。"))
    # 零幅度会得到 Inf；仅在被使用的频段内才会被判定为错误。
    il = -20 .* log10.(abs.(ComplexF64.(sdd21)))
    return cei_ild_from_il(f_hz, il; kwargs...)
end

"""
    sdd21_from_sparams(S; input_pair, output_pair, z0_ohm)

S 形状为 (4,4,N)，S[接收端口,激励端口,频点]，值为线性复数。
input_pair=(输入P,输入N)，output_pair=(输出P,输出N)，端口从 1 开始。
仅接受所有单端端口均为实数 50 ohm 参考阻抗；不在此函数中重归一化。
"""
function sdd21_from_sparams(S::AbstractArray{<:Number,3};
                            input_pair::Tuple{Int,Int},
                            output_pair::Tuple{Int,Int}, z0_ohm)
    size(S, 1) == 4 && size(S, 2) == 4 || throw(DimensionMismatch("S 必须为 4×4×N。"))
    ports = [input_pair..., output_pair...]
    sort(ports) == [1, 2, 3, 4] || throw(ArgumentError("四个 P/N 端口必须恰好覆盖 1:4，不能重复。"))
    z = z0_ohm isa Number ? fill(ComplexF64(z0_ohm), 4) : ComplexF64.(collect(z0_ohm))
    length(z) == 4 && all(isfinite, z) && all(abs.(z .- 50) .<= 1e-8) ||
        throw(ArgumentError("请先把四个单端端口重归一化到实数 50 ohm；本函数不会静默变更阻抗。"))
    ip, im = input_pair
    op, om = output_pair
    return ComplexF64[(S[op,ip,k] - S[op,im,k] - S[om,ip,k] + S[om,im,k]) / 2
                       for k in axes(S, 3)]
end

function cei_ild_sparams(f_hz::AbstractVector{<:Real}, S::AbstractArray{<:Number,3};
                         input_pair::Tuple{Int,Int}, output_pair::Tuple{Int,Int},
                         z0_ohm, kwargs...)
    length(f_hz) == size(S, 3) || throw(DimensionMismatch("S 的第三维必须等于频率点数。"))
    sdd = sdd21_from_sparams(S; input_pair=input_pair, output_pair=output_pair, z0_ohm=z0_ohm)
    return cei_ild(f_hz, sdd; kwargs...)
end

"""
    read_s4p(path)

留给用户的 Touchstone 读取接口，不含解析器，不能直接读真实文件。
接入的读取函数应返回：
    (f_hz = Vector{Float64}, S = Array{ComplexF64,3}, z0_ohm = 50.0)
S 的尺寸必须是 4×4×N。读取端应处理 Hz/kHz/MHz/GHz、RI/MA/DB、角度制、
Touchstone 行列顺序；z0_ohm 必须来自真实文件，不能不检查就写死为 50。
"""
function read_s4p(path::AbstractString)
    throw(ArgumentError("S4P 读取接口尚未接入：$path。请通过 reader=my_reader 传入读取函数，返回 (f_hz, S, z0_ohm)。"))
end

function cei_ild_s4p(path::AbstractString; reader=read_s4p,
                      input_pair::Tuple{Int,Int}, output_pair::Tuple{Int,Int}, kwargs...)
    data = reader(path)
    all(k -> hasproperty(data, k), (:f_hz, :S, :z0_ohm)) ||
        throw(ArgumentError("reader 必须返回含 f_hz、S、z0_ohm 的 NamedTuple。"))
    return cei_ild_sparams(data.f_hz, data.S; input_pair=input_pair,
        output_pair=output_pair, z0_ohm=data.z0_ohm, kwargs...)
end

function print_ild_summary(r; io::IO=stdout)
    println(io, "Profile: ", r.settings.profile_name)
    @printf(io, "FOM_ILD = %.9f dB\n", r.fom_ild_db)
    @printf(io, "Unweighted ILD RMS = %.9f dB (FOM band)\n", r.ild_rms_unweighted_db)
    @printf(io, "Baud rate = %.6f GBd; ft = %.6f GHz; fr = %.6f GHz\n",
        r.settings.fb_hz / 1e9, r.settings.ft_hz / 1e9, r.settings.fr_hz / 1e9)
    @printf(io, "Fit band = %.6f to %.6f GHz; N = %d\n",
        first(r.f_fit_hz) / 1e9, last(r.f_fit_hz) / 1e9, r.n_fit)
    @printf(io, "FOM band = %.6f to %.6f GHz; N = %d\n",
        first(r.f_fom_hz) / 1e9, last(r.f_fom_hz) / 1e9, r.n_fom)
    println(io, "Coefficients (a0,a1,a2,a4): ", r.coefficients)
    println(io, "Bounds applied: ", r.bounds_applied, "; grid: ", r.settings.grid_mode)
    println(io, "Coarse sampling: ", r.coarse_sampling)
    println(io, "User-limit comparison (<=): ", r.meets_user_limit)
    println(io, "This calculation does not certify CEI-56G-VSR-NRZ compliance.")
end

"""导出两张 CSV 曲线表和一个参数/结果摘要；prefix 可包含目录。"""
function write_ild_csv(prefix::AbstractString, r)
    mkpath(dirname(abspath(prefix)))
    fitpath, fompath, summarypath = prefix * "_fit.csv", prefix * "_fom.csv", prefix * "_summary.txt"
    open(fitpath, "w") do io
        println(io, "f_Hz,f_GHz,IL_dB,ILfit_dB,ILD_dB")
        for i in eachindex(r.f_fit_hz)
            @printf(io, "%.17g,%.17g,%.17g,%.17g,%.17g\n", r.f_fit_hz[i], r.f_fit_hz[i]/1e9,
                r.il_db[i], r.il_fit_db[i], r.ild_db[i])
        end
    end
    open(fompath, "w") do io
        println(io, "f_Hz,ILD_dB,W,W_times_ILD_squared,cumulative_FOM_dB")
        for i in eachindex(r.f_fom_hz)
            @printf(io, "%.17g,%.17g,%.17g,%.17g,%.17g\n", r.f_fom_hz[i], r.ild_fom_db[i],
                r.weights[i], r.weighted_ild_squared[i], r.cumulative_fom_db[i])
        end
    end
    open(summarypath, "w") do io
        print_ild_summary(r; io=io)
        println(io, "\nAll settings:\n", r.settings)
        println(io, "\nCoefficient fixing history:\n", r.fit_history)
    end
    return (fit_csv=fitpath, fom_csv=fompath, summary=summarypath)
end

end # module
