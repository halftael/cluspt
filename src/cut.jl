using DataFrames
using LinearAlgebra
using CSV
using Clustering
using ProgressMeter
using GLMakie

GLMakie.activate!()


# load data
A = CSV.read("../data/adj.csv",DataFrame)
D = CSV.read("../data/degree.csv",DataFrame)

names = A[!,:name][:]



#  Normal Laplace Matrix
function Laplace(D,A)
   
  A = select(A,Not(:name))
  D = select(D,Not(:name))

  # construct Laplace Matrix
  A = Matrix{Float64}(A) 
  D = Matrix{Float64}(D) 
  L = D .- A

  ## normalize
  vals,vecs = eigen(D)
  sqrt_vals = sqrt.(vals)
  D_sqrt = vecs * Diagonal(sqrt_vals) * vecs'
  L = inv(D_sqrt) * L * inv(D_sqrt) 
  
  return L
end



# dimensions reduction 
## feature
L = Laplace(D,A)
λ,U = eigen(L)

## Elbow Method
function elbow(λ)

  xr = collect(1:length(λ))
  yr = sort(real(λ),rev=true)

  fig = Figure()
  ax = Axis(fig[1,1])
  lines!(ax,xr,yr)
  
  return fig
end

## k dimensions
function Feature(λ,U,k) 
  k = k
  indexs = findall(x->real(x)>=sort(real(λ),rev=true)[k],λ)
  F = begin
    ini = indexs[1]
    M = real(U[:,ini])
    for i in indexs[2:end]
      M = hcat(M,real(U[:,i]))
    end
    M
  end 
  return F
end

# kmeans
## Elbow Method
k = 4300
F = Feature(λ,U,k)
result = kmeans(F,k; maxiter=200)



