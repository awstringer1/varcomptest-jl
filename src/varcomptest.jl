module varcomptest

## Dependencies ----
using 
  Random,
  Distributions, 
  DataFrames, 
  LinearAlgebra, 
  CategoricalArrays, 
  SparseArrays, 
  LoopVectorization,
  StatsModels,
  MixedModels,
  PrettyTables,
  Printf,
  UnicodePlots,
  Statistics
  # Regex

## END dependencies ----

## Data structrures ----

# Define the model #
mutable struct Model
  m::Int64
  p::Int64
  n::Int64
  mvec::Vector{Int64}
  ZR::SparseMatrixCSC{Float64, Int64}
  ZRp::SparseMatrixCSC{Float64, Int64}
  Zqr::SparseArrays.SPQR.QRSparse{Float64, Int64}
  Xqr::LinearAlgebra.QRCompactWY{Float64, Matrix{Float64}, Matrix{Float64}}
  Qx::Matrix{Float64}
  PztQx::Matrix{Float64}
  Pzty::Vector{Float64}
  PztQxLp::Matrix{Float64}
  PztyLp::Vector{Float64}
  Utysqnorm::Float64
  Z::SparseMatrixCSC
  X::Matrix{Float64}
  y::Vector{Float64}
  idxlist::Vector{UnitRange{Int64}}
  Ldecomp::SparseArrays.CHOLMOD.Factor{Float64, Int64}
  HitHj::Vector{Matrix{Float64}}
  FitFj::Vector{Matrix{Float64}}
end;

function Model(y::Vector{Float64}, X::Matrix{Float64}, Z::SparseMatrixCSC{Float64, Int64}, mvec::Vector{Int64})
  Zqr = qr(Z);
  ZR = Zqr.R;
  Xqr = qr(X);
  N, p = size(X);
  Qx = Xqr.Q * I(p);
  PztQx = Zqr.Q' * Qx[Zqr.prow, :];
  Pzty = Zqr.Q' * y[Zqr.prow];
  Utysqnorm = sum(abs2, (Xqr.Q' * y)[(p + 1):N]);
  idxlist = [sum(mvec[1:i-1]) .+ (1:mvec[i]) for i in eachindex(mvec)];
  # Precompute the Cholesky factor
  ZRDR = ZR * ZR';
  Ldecomp = cholesky(Symmetric(ZRDR), shift = 1.);
  ZRp = ZR[Ldecomp.p, invperm(Zqr.pcol)];
  PztQxLp = PztQx;
  r = size(Ldecomp.L, 1);
  PztQxLp[1:r, :] = PztQxLp[1:r, :][Ldecomp.p, :]
  PztyLp = Pzty;
  PztyLp[1:r] = PztyLp[1:r][Ldecomp.p]
  # Preallocate the dense matrices required for the Hessian
  d = length(mvec)
  HitHj = Vector{Matrix{Float64}}(undef, Int64(d * (d + 1) / 2))
  FitFj = Vector{Matrix{Float64}}(undef, Int64(d * (d + 1) / 2))
  idx = 1
  for i in 1:d
    for j in i:d
      HitHj[idx] = zeros(mvec[i], mvec[j])
      FitFj[idx] = zeros(mvec[i], mvec[j])
      idx += 1
    end
  end

  return Model(
    size(Z, 2),
    size(X, 2),
    size(X, 1),
    mvec,
    ZR,
    ZRp,
    Zqr,
    Xqr,
    Qx,
    PztQx,
    Pzty,
    PztQxLp,
    PztyLp,
    Utysqnorm,
    Z,
    X,
    y,
    idxlist,
    Ldecomp,
    HitHj,
    FitFj
  );

end;

function Model!(model::Model, newy::AbstractVector{<:Float64})
  # Update a given model
  model.Pzty .= model.Zqr.Q' * newy[model.Zqr.prow];
  model.Utysqnorm = sum(abs2, (model.Xqr.Q' * newy)[(model.p + 1):model.n]);
  r = size(model.Ldecomp.L, 1)
  model.PztyLp .= model.Pzty
  model.PztyLp[1:r] .= model.PztyLp[1:r][model.Ldecomp.p]
end

# Optimization control and printing #
struct NewtonControl
  eps::Float64
  maxitr::Int64
  kappa::Float64
  verbose::Bool
end

function NewtonControl(; 
  eps::Float64 = 1e-06,
  maxitr::Int64 = 100,
  kappa::Float64 = 1e-03,
  verbose::Bool = false
)
  return NewtonControl(eps, maxitr, kappa, verbose)
end

struct optResults
  init::Vector{Float64}
  par::Vector{Float64}
  val::Float64
  derivs::NamedTuple{(:gradient, :Hessian), Tuple{Vector{Float64}, Matrix{Float64}}}
  itr::Int64
  control::NewtonControl
  executiontime::Float64
  A::Union{Nothing, Matrix{Float64}}
end
# Print method for optimization results
function Base.show(io::IO, x::optResults)
  indent = "    "
  if x.A == nothing
    line = "------------------------------------------"
    println(io, line)
    println(io, "Unconditional Optimization of 𝛕")
    println(io, line)
    println(io, indent, "Starting value: ", round.(x.init, digits = 3))
    println(io, indent, "Minimizer: ", round.(x.par, digits = 3))
    println(io, indent, "Minimum value: ", round(x.val, digits = 3))
    println(io, indent, "Number of iterations: ", x.itr)
    println(io, indent, "Execution time: ", round(x.executiontime, digits = 3), " seconds")
    println(io, line)
    println(io, "Derivative information:")
    println(io, line)
    println(io, indent, "Gradient: ", round.(x.derivs.gradient, digits=Int64(round(log10(x.control.eps)))))
    E = eigen(x.derivs.Hessian)
    println(io, indent, "Hessian Eigenvalues: ", round.(E.values, digits = 3))
    println(io, line)
    println(io, "Control parameters: ")
    println(io, line)
    println(io, indent, "Optimization tolerance: ", round(x.control.eps, digits = 3))
    println(io, indent, "Maximum number of iterations: ", x.control.maxitr)
    println(io, indent, "Eigenvalue correction (κ): ", x.control.kappa)
    println(io, line)
  else
    r, d = size(x.A)
    Aqr = qr(x.A')
    Q2 = (Aqr.Q * I(d))[:, (r + 1):d]
    line = "----------------------------------------------"
    println(io, line)
    println(io, "Conditional Optimization of 𝛕 such that A𝛕 = 0 where A = ", x.A)
    println(io, line)
    println(io, indent, "Unprojected Minimizer: ", round.(x.par, digits = 3))
    println(io, indent, "Projected Minimizer: ", round.(Q2' * x.par, digits = 3))
    println(io, indent, "Minimum value: ", round(x.val, digits = 3))
    println(io, indent, "Number of iterations: ", x.itr)
    println(io, indent, "Execution time: ", round(x.executiontime, digits = 3), " seconds")
    println(io, line)
    println(io, "Derivative information:")
    println(io, line)
    println(io, indent, "Unprojected Gradient: ", round.(Q2 * x.derivs.gradient, digits=Int64(round(abs(log10(x.control.eps))))))
    println(io, indent, "Projected Gradient: ", round.(x.derivs.gradient, digits=Int64(round(abs(log10(x.control.eps))))))
    E = eigen(x.derivs.Hessian)
    Ec = eigen(Q2 * x.derivs.Hessian * Q2')
    println(io, indent, "Unprojected Hessian Eigenvalues: ", round.(Ec.values, digits = 3))
    println(io, indent, "Projected Hessian Eigenvalues: ", round.(E.values, digits = 3))
    println(io, line)
    println(io, "Control parameters: ")
    println(io, line)
    println(io, indent, "Optimization tolerance: ", x.control.eps)
    println(io, indent, "Maximum number of iterations: ", x.control.maxitr)
    println(io, indent, "Eigenvalue correction (κ): ", x.control.kappa)
    println(io, line)
  end
end

struct FixedEffects
  names::Vector{String}
  beta::Vector{Float64}
  covmat::Matrix{Float64}
end

function Base.show(io::IO, x::FixedEffects)
  column_labels = ["Coefficient", "Std. Error", "z", "Pr(>|z|)"]
  beta = x.beta
  stderr = sqrt.(diag(x.covmat))
  zvals = beta ./ stderr
  pvals = 2 .* (1 .- cdf.(Normal(), abs.(zvals)))

  pval_highlight_green = TextHighlighter(
    (data, i, j) -> (j == 4) && data[i, j] <= .05,
    crayon"green bold"
  )

  data = hcat(beta, stderr, zvals, pvals)
  style = TextTableStyle(first_line_column_label = crayon"bold");
  table_format = TextTableFormat(borders = text_table_borders__unicode_rounded);


  pretty_table(data;
        column_labels = column_labels,
        row_labels = x.names,
        style = style,
        table_format = table_format,
        formatters = [fmt__round(4)],
        highlighters = [pval_highlight_green]
    )

end

struct VarianceComponents
  names::Vector{String}
  tau::Vector{Float64}
  ftable::Matrix{Float64}
  A::Union{Nothing, Matrix{Float64}}
  lrtboot::Union{Nothing, Vector{Float64}}
  pval::Union{Nothing, Float64}
  pvaloneside::Union{Nothing, Float64}
  lrtcond::Union{Nothing, Float64}
end

function Base.show(io::IO, x::VarianceComponents)
  column_labels = ["Component", "Df", "Sum Sq", "Mean Sq", "F value", "Pr(>F)"]
  tau = x.tau
  d = length(tau) - 1
  ftable = x.ftable
  # Add a row of zeroes for the residual variance

  tau_highlight_red = TextHighlighter(
    (data, i, j) -> (j == 1 && i != d + 1) && data[i, j] <= 0.,
    crayon"red bold"
  )
  pval_highlight_green = TextHighlighter(
    (data, i, j) -> (j == 7 && i != d + 2) && data[i, j] <= .05,
    crayon"green bold"
  )

  data = hcat(tau, ftable)
  style = TextTableStyle(first_line_column_label = crayon"bold");
  table_format = TextTableFormat(borders = text_table_borders__unicode_rounded);
  rownames = vcat(x.names[1:d], "Residual")

  println(io, "F-tests of variance components:")
  pretty_table(data;
      column_labels = column_labels,
      row_labels = rownames,
      style = style,
      table_format = table_format,
      formatters = [fmt__round(4), (v, i , j) -> (i == d + 1 && j > 4) ? "-" : v],
      highlighters = [tau_highlight_red, pval_highlight_green]
    )
end

struct bootResults
  # The sampled data
  samples::Matrix{Float64}
  # The sampled log-likelihood ratios
  lrt::Vector{Float64}
  # The sampled MLE
  mle::Union{Vector{Float64}, Matrix{Float64}}
  # Number of samples
  B::Int64
  # p-values
  pval::Float64
  pvaloneside::Float64
  A::Union{Nothing, Matrix{Float64}}
end

function Base.show(io::IO, x::bootResults)
  B = x.B
  d = size(x.mle, 2)
  println("Results based on $B bootstrap samples with $d parameters.")
  println("")
  println("Distribution of the MLE")
  mle = DataFrame(x.mle, ["𝛕" * "$i" for i in 1:d])
  tbl = describe(mle)
  style = TextTableStyle(first_line_column_label = crayon"bold");
  table_format = TextTableFormat(borders = text_table_borders__unicode_rounded);
  pretty_table(tbl; style = style, table_format = table_format)
  println("Access individual samples using 'mle' property.")
  println("")
  if (x.A != nothing)
    r = size(x.A, 1)
    if r < d
      println("Distribution of A𝛕 where A = ", x.A)
      mleA = DataFrame(x.mle * x.A', ["A𝛕" * "$i" for i in 1:r])
      tbl = describe(mleA)
      pretty_table(tbl; style = style, table_format = table_format)
    end
  end
  println("")
  println("Bootstrap LRT distribution")
  print(
    UnicodePlots.histogram(
      x.lrt, vertical = true, width = 50, nbins = B / 4.,
      title = "Bootstrap LRT distribution, $B samples:", xlabel = "Bootstrapped LRT", ylabel = "Frequency"
    )
  )
  println("")
  tbldat = [mean(x.lrt), x.pval, x.pvaloneside]
  tblnames = ["Mean LRT", "BS p-val, A𝛕 ≠ 0", "BS p-val, A𝛕 > 0"]
  pval_highlight_green2 = TextHighlighter(
    (data, i, j) -> (j != 1) && data[i, j] <= .05,
    crayon"green bold"
  )
  pretty_table(tbldat';
    column_labels = tblnames,
    style = style,
    table_format = table_format,
    formatters = [fmt__round(4)],
    highlighters = [pval_highlight_green2]
  )
end

struct VarCompModel
  opt::optResults
  optcond::Union{Nothing, optResults}
  fe::FixedEffects
  vr::VarianceComponents
  model::Model
  bootresults::Union{Nothing, bootResults}
end

function Base.show(io::IO, x::VarCompModel)
  println("Variance components model fit by maximum normalized residual likelihood.")
  println("")
  println("Fixed Effects Estimates")
  show(io, x.fe)
  println("")
  show(io, x.vr)
  println("")
  if x.bootresults != nothing
    show(io, x.bootresults)
  end
  println("")
  println("Access optimization information through 'opt' and 'optcond' properties.")
end

struct VarCompControl
  newtoncontrol::NewtonControl
  B::Int64 ## Bootstrap samples
end


## END Data structures ----

## Likelihood and derivatives ----
# Twice the negative normalized residual log-likelihood
nrll = function(tau::Vector{Float64}, model::Model)
  taurep = vcat([fill(tau[i], model.mvec[i]) for i in eachindex(tau)]...);
  Dtau = spdiagm(0 => taurep[model.Zqr.pcol]);
  ZRDR = model.ZR * Dtau * model.ZR';
  Ldecomp = model.Ldecomp
  # Update the cholesky factor
  Ldecomp = cholesky!(Ldecomp, Symmetric(ZRDR), shift = 1., check = false);
  if !issuccess(Ldecomp) 
    return NaN 
  end
  r = size(Ldecomp.L, 1);
  PztQx = copy(model.PztQx);
  PztQx[1:r, :] .= Ldecomp.L \ model.PztQxLp[1:r, :];
  QtAinvQ = PztQx' * PztQx;
  Pzty = copy(model.Pzty);
  Pzty[1:r] = Ldecomp.L \ model.PztyLp[1:r];
  ytSinvy = sum(abs2, Pzty);
  QxtAinvy = PztQx' * Pzty;
  quadform = (ytSinvy - QxtAinvy' * (QtAinvQ \ QxtAinvy)) / model.Utysqnorm;
  logdet = LinearAlgebra.logdet(Ldecomp) + LinearAlgebra.logdet(QtAinvQ);
  logdet + Float64(model.n - model.p) * log(quadform);
end;

# Gradient and Hessian of twice the negative normalized residual log-likelihood
function nrllD(tau::Vector{Float64}, model::Model) 
  d = length(tau)
  M = sum(model.mvec)

  # expand tau into repeated entries (length M)
  taurep = vcat([fill(tau[i], model.mvec[i]) for i in eachindex(tau)]...)
  Dtau = Diagonal(taurep[model.Zqr.pcol])
  ZRDR = model.ZR * Dtau * model.ZR'
  Ldecomp = model.Ldecomp
  # Update the cholesky factor
  Ldecomp = cholesky!(Ldecomp, Symmetric(ZRDR), shift = 1., check = false);

  m = size(Ldecomp.L, 1)   # number of rows in the factor (same as model$m)

  # ---- term1 = diag( (L^{-1} * ZR)' * (L^{-1} * ZR) )  (length M)
  ## Half of the time and most of the memory allocation; large sparse solve ##
  LinvRz = Ldecomp.L \ model.ZRp # Checked                # m x M

  term1 = vec(sum(abs2, LinvRz; dims = 1))      # length M

  # ---- QtAinvQ and its factor
  PztQx = copy(model.PztQx)                      # don't mutate model
  PztQx[1:m, :] = Ldecomp.L \ model.PztQxLp[1:m, :]
  
  QtAinvQ = PztQx' * PztQx # Checked                       # p x p
  Qdecomp = cholesky(Symmetric(QtAinvQ))         # factor

  # ---- term2 = diag( (Qdecomp.L \ (PztQx[1:m,:]' * LinvRz))^2 ) length M
  QxSinvZfact = PztQx[1:m, :]' * LinvRz          # p x M
  CinvG = Qdecomp.L \ QxSinvZfact                # p x M
  term2 = vec(sum(abs2, CinvG; dims = 1))        # length M

  diagvec = term1 .- term2                       # length M

  # ---- group sums for logdet derivative: replicate R's
  prefix_diag = vcat(0., cumsum(diagvec))   # length M+1
  ivec = vcat(0, cumsum(model.mvec)) .+ 1        # indices into prefix_diag, length d+1
  logdeterm = diff(prefix_diag[ivec])            # length d
  # logdeterm CORRECT, checked

  # ---- quadform and its pieces
  Pzty = copy(model.Pzty)
  Pzty[1:m] = Ldecomp.L \ model.PztyLp[1:m]
  ytSinvy = sum(abs2, Pzty) # Correct
  QxtAinvy = PztQx' * Pzty # Correct
  quadform = (ytSinvy - QxtAinvy' * (QtAinvQ \ QxtAinvy))

  # ---- bvec
  ZtSinvy = LinvRz' * Pzty[1:m] # Correct                 # length M
  temp = PztQx' * Pzty # Correct                           # length p
  sol = Qdecomp.L \ temp # Correct                         # length p (solve L * sol = temp)
  ZtSinvQmidQtSinvy = CinvG' * sol # Correct               # length M
  bvec = ZtSinvy .- ZtSinvQmidQtSinvy             # length M

  prefix_b2 = vcat(0., cumsum(bvec.^2))     # length M+1
  qfterm = diff(prefix_b2[ivec]) ./ quadform     # length d

  grad = logdeterm .- Float64(model.n - model.p) .* qfterm
  # ---- Hessian
  hterm3 = qfterm * qfterm'                       # d x d

  # build block-expanded matrix of bvec (M x d) like R's bexpand
  bexpand = zeros(M, d)
  idx = 1
  for j in 1:d
      len = model.mvec[j]
      bexpand[idx:idx+len-1, j] .= bvec[idx:idx+len-1]
      idx += len
  end

  A = LinvRz * bexpand    # m x d
  B = CinvG * bexpand     # p x d
  hterm2 = (A' * A .- B' * B) ./ quadform     # d x d

  # logdet Hessian term (double loop over blocks)
  hterm1 = zeros(d, d)
  starti = 1
  idx = 1
  for i in 1:d
      li = model.mvec[i]
      Fi = LinvRz[:, starti:starti+li-1]    # m x li
      Hi = CinvG[:, starti:starti+li-1]     # p x li
      startj = starti
      for j in i:d
          lj = model.mvec[j]
          Fj = LinvRz[:, startj:startj+lj-1]
          Hj = CinvG[:, startj:startj+lj-1]
          HitHj = model.HitHj[idx]
          mul!(HitHj, Hi', Hj, -1., 0.)
          FitFj = Fi' * Fj
          rows, cols, vals = findnz(FitFj) # TODO: precompute these
          idx2 = 1
          @tturbo for k in eachindex(rows)
            HitHj[rows[k], cols[k]] += FitFj.nzval[k]
          end
          val = sum(abs2, HitHj)
          hterm1[i, j] = -val
          hterm1[j, i] = -val
          startj += lj
          idx +=1 
      end
      starti += li
  end
  # hterm1 CORRECT, checked

  hess = hterm1 .+ Float64(model.n - model.p) .* (2 .* hterm2 .- hterm3)

  return hcat(grad, hess)
end;

## END Likelihood and derivatives ----

## Optimization ----

newton = function(tau::Vector{Float64}, model::Model, control::NewtonControl; A::Union{Nothing, Matrix{Float64}} = nothing)
  verbose = control.verbose
  tauinit = copy(tau)
  d = length(tau)
  if A == nothing
    Q2 = I(d)
  else
    r = size(A, 1)
    Aqr = qr(A')
    Q2 = (Aqr.Q * I(d))[:, (r + 1):d]
  end
  tauConstr = Q2' * tau
  tau .= Q2 * tauConstr
  r = length(tauConstr)
  eigtol = control.kappa * 2.
  
  D = nrllD(tau, model)
  gg = Q2' * D[:, 1]
  H = Q2' * D[:, 2:(d + 1)] * Q2
  E = eigen(H)
  stepvec = zeros(r)
  proposed = zeros(r)
  
  itr = 0
  converged = maximum(abs.(gg)) < control.eps || itr >= control.maxitr
  t = @elapsed begin 
    while !converged
      itr = itr + 1
      if verbose
        println("Iteration $itr of Newton")
      end
      # Eigendecomposition
      E = eigen(H)
      if verbose
        println("Hessian eigenvalues: ", round.(E.values, digits = 3))
        println("Adjusted eigenvalues: ", round.(abs.(E.values) .+ control.kappa, digits = 3))
      end
      H .= E.vectors * diagm(abs.(E.values) .+ control.kappa) * E.vectors'

      stepvec .= .-H \ gg
      proposed .= tauConstr .+ stepvec
      # Step halving
      good = false
      numstephalve = 0
      while !good
        newval = nrll(Q2 * proposed, model)
        if isnan(newval)
          proposed .= proposed ./ 2.
          numstephalve = numstephalve + 1
        else
          good = true
        end
      end
      tauConstr .= proposed
      tau .= Q2 * tauConstr
      D .= nrllD(tau, model)
      gg .= Q2' * D[:, 1]
      H .= Q2' * D[:, 2:(d + 1)] * Q2
      if verbose
        println("New tau: ", round.(tau, digits = 3))
        println("New gradient: ", round.(gg, digits = 3))
      end
      if maximum(abs.(gg)) < control.eps || itr >= control.maxitr 
        converged = true
      end
    end
  end
  out = optResults(
    tauinit,
    tau,
    nrll(tau, model),
    (gradient = gg, Hessian = H),
    itr,
    control,
    t,
    A
  )
  return out
end

## END Optimization ----

## Fit the model ----

function varcompmodel(
  formula::FormulaTerm,
  dat::DataFrame;
  control::VarCompControl = VarCompControl(NewtonControl(), 0),
  A::Union{Nothing, Matrix{Float64}}=nothing
)
  # Parse the formula and create the model matrices
  # MixedModels sorts the variables lexicographically which messes up ANOVA and also my custom LRT stuff
  # So parse the formula and build the matrices manually
  rhs_string = string.(formula.rhs)
  rhs_string = isa(rhs_string, Tuple) ? rhs_string : [rhs_string]
  reterms = filter(s -> occursin("|", s), rhs_string)
  renames = [match(r"\|\s*(.+)\)", s).captures[1] for s in reterms]

  lhs = formula.lhs
  rhs_terms = isa(formula.rhs, AbstractVector) ? formula.rhs : [formula.rhs]
  rand_terms = [term(1) | Term(Symbol(r)) for r in renames]
  fixed_terms = []
  
  for t in rhs_terms
      if !occursin("|", string(t))
          push!(fixed_terms, t)
      end
  end
  if !any([string(t) == "1" for t in fixed_terms])
    insert!(fixed_terms, 1, term(1))
  end

  d = length(rand_terms)
  p = length(fixed_terms)

  formulas = [lhs ~ (sum(fixed_terms) + r) for r in rand_terms]
  Zblocks = Vector{Adjoint{Float64, SparseArrays.SparseMatrixCSC{Float64, Int64}}}(undef, d)

  lmod = nothing
  for i in 1:d
    lmod = LinearMixedModel(formulas[i], dat) # Does NOT fit, just creates quantities
    Zblocks[i] = sparse(lmod.reterms[1])
  end

  N, p = size(lmod.Xymat); p -= 1
  X = lmod.Xymat[:, 1:p]
  y = lmod.Xymat[:, p + 1]
  Z = sparse(hcat(Zblocks...));
  mvec = [size(block, 2) for block in Zblocks];

  ## First: ANOVA ----
  ftable = zeros(d + 1, 5)
  XZ0 = sparse(X)
  qr0 = nothing
  r0 = nothing
  ss0 = nothing
  qr1 = nothing
  r1 = nothing
  ss1 = nothing
  ranktol = 1e-08
  M = zeros(d, d)
  S = zeros(d)
  rvec = zeros(d)
  for j in 1:d
    qr0 = qr(XZ0)
    r0 = sum(abs.(diag(qr0.R)) .> ranktol) # Rank
    ss0 = sum(abs2, (qr0.Q' * y[qr0.prow])[(r0 + 1):N])

    XZ1 = hcat(XZ0, sparse(Zblocks[j]))
    qr1 = qr(XZ1)
    r1 = sum(abs.(diag(qr1.R)) .> ranktol) # Rank

    ss1 = sum(abs2, (qr1.Q' * y[qr1.prow])[(r1 + 1):N])

    rvec[j] = r1 - r0

    if r1 > r0
      ftable[j, 1:3] = [r1 - r0, ss0 - ss1, (ss0 - ss1) / (r1 - r0)]
      # Fill the M matrix
      M[j, j] = sum(abs2, (qr0.Q' * Matrix(Zblocks[j])[qr0.prow, :])[(r0 + 1):N, :])
      for i in (j + 1):d
        M[j, i] = sum(abs2, (qr0.Q' * Matrix(Zblocks[i])[qr0.prow, :])[(r0 + 1):N, :]) - sum(abs2, (qr1.Q' * Matrix(Zblocks[i])[qr1.prow, :])[(r1 + 1):N, :])
      end
    else
      @error "Some sequential ANOVA terms had zero or negative degrees of freedom. Expect an error."
    end
    XZ0 = XZ1
  end
  # Residuals
  ftable[d + 1, 1:3] = [N - r1, ss1, ss1 / (N - r1)]
  ftable[1:d, 4] = ftable[1:d, 3] / ftable[d + 1, 3]
  for i in 1:d
    ftable[i, 5] = 1. - cdf(FDist(ftable[i, 1], ftable[d + 1, 1]), ftable[i, 3] / ftable[d + 1, 3])
  end
  sigmasqest = ftable[d + 1, 3]
  # Initial values
  S = ftable[1:d, 2] / sigmasqest - rvec
  tauinit = M \ S
  tauinit = [t >= 0 ? t : 0. for t in tauinit]

  # Create the model
  model = Model(y, X, Z, mvec);
  
  # Fit the model
  # tauinit = Float64.(zeros(d))
  opt = newton(tauinit, model, control.newtoncontrol);
  tauopt = copy(opt.par)
  
  # Fixed effects
  taumle = copy(opt.par)
  taurep = vcat([fill(taumle[i], model.mvec[i]) for i in eachindex(taumle)]...)
  Dtau = Diagonal(taurep[model.Zqr.pcol])
  ZRDR = model.ZR * Dtau * model.ZR'
  Ldecomp = model.Ldecomp
  Ldecomp = cholesky!(Ldecomp, Symmetric(ZRDR), shift = 1., check = false);
  PztX = model.Zqr.Q' * model.X[model.Zqr.prow, :]
  Pzty = model.Zqr.Q' * model.y[model.Zqr.prow]
  m = size(Ldecomp.L, 1)
  PztX[1:m, :] = Ldecomp.L \ PztX[1:m, :][Ldecomp.p, :]
  Pzty[1:m] = Ldecomp.L \ Pzty[1:m][Ldecomp.p]
  betaest = (PztX' * PztX) \ (PztX' * Pzty)
  
  # Residual standard deviation
  PztQx = model.Zqr.Q' * model.Qx[model.Zqr.prow, :]
  PztQx[1:m, :] = Ldecomp.L \ PztQx[1:m, :][Ldecomp.p, :]
  QtAinvQ = PztQx' * PztQx
  QxtAinvy = PztQx' * Pzty
  sigmasqest = ( Pzty' * Pzty - QxtAinvy' * (QtAinvQ \ QxtAinvy) ) / Float64(model.n - model.p)

  betacovmat = sigmasqest * inv(PztX' * PztX)

  fe = FixedEffects(lmod.feterm.cnames, betaest, betacovmat)
  
  # Variance components estimates
  varcompest = copy(taumle)
  push!(varcompest, sigmasqest)


  ## Fit the conditional model ----
  docond = false
  optcond = nothing
  optcondval = 0
  if A != nothing
    if size(A, 1) < d
      optcond = newton(tauopt, model, control.newtoncontrol, A = A);
      tauoptcond = copy(optcond.par)
      optcondval = copy(optcond.val)
      docond = true
    else
      # A is full rank so the conditional MLE is zero.
      # In this case we still bootstrap the full-zero hypothesis
      A = Matrix(Float64.(I(d)))
    end
  end

  ## Bootstrapping ----
  B = control.B
  lrtboot = nothing
  pval = -1.
  pvaloneside = -1.
  boot = nothing
  if B > 0
    lrtboot = zeros(B)
    pvalind = zeros(B)
    pvalonesideind = zeros(B)
    mleboot = zeros(B, d)
    # Obtain the model quantities under the conditional model
    Zsamp = zeros(N, B)
    Zsamp = randn!(Zsamp)
    condmle = docond ? copy(optcond.par) : zeros(d)
    taumle = copy(opt.par)
    if docond
      # Adjust the samples
      taurep = vcat([fill(condmle[i], model.mvec[i]) for i in eachindex(condmle)]...)
      Dtau = Diagonal(taurep[model.Zqr.pcol])
      ZRDR = model.ZR * Dtau * model.ZR'
      Ldecomp = model.Ldecomp
      Ldecomp = cholesky!(Ldecomp, Symmetric(ZRDR), shift = 1., check = false);
      r = size(Ldecomp.L, 1)
      Zsamp[1:r, :] .= sparse(Ldecomp.L) * Zsamp[1:r, :][invperm(Ldecomp.p), :]
      Zsamp .= (model.Zqr.Q * Zsamp)[invperm(model.Zqr.prow), :]
    end
    ysamp = similar(y)
    modelsamp = Model(y, X, Z, mvec) # Create it again, for now
    for b in 1:B
      ysamp = @view Zsamp[:, b]
      Model!(modelsamp, ysamp)
      taumle = copy(opt.par)
      optsamp = newton(taumle, modelsamp, control.newtoncontrol, A = nothing);
      if docond
        taumle = copy(optcond.par)
        optsampcond = newton(taumle, modelsamp, control.newtoncontrol, A = A);
        lrtboot[b] = -optsamp.val + optsampcond.val
        pvalind[b] = lrtboot[b] >= -opt.val + optcond.val
        pvalonesideind[b] = lrtboot[b] >= -opt.val + optcond.val && all(A * optsamp.par .>= 0.)
      else
        lrtboot[b] = -optsamp.val
        pvalind[b] = -optsamp.val >= -opt.val
        pvalonesideind[b] = -optsamp.val >= -opt.val && all(optsamp.par .>= 0.)
      end
      mleboot[b, :] = optsamp.par
    end
    pval = mean(pvalind)
    pvaloneside = mean(pvalonesideind)
    boot = bootResults(Zsamp, lrtboot, mleboot, B, pval, pvaloneside, A)
  end
  vr = VarianceComponents(
    renames, varcompest, ftable, 
    A == nothing ? I(d) : A, 
    lrtboot, pval, pvaloneside, 
    optcond == nothing ? -opt.val : -opt.val + optcond.val
  )

  
  out = VarCompModel(opt, optcond, fe, vr, model, boot)

  return out
end

## END fit the model ----


## Exports ----

export varcompmodel
export FixedEffects
export VarCompControl
export NewtonControl

## END Exports ----


end # module varcomptest
