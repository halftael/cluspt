using CSV
using DataFrames
using LinearAlgebra
using ProgressMeter

#load dotaSet
#dataSet = CSV.read("../data/SC.csv",DataFrame)
#system_name = CSV.read("../data/system_name.csv",DataFrame)


# features and system name of SGD
function toSystem(dataSet::DataFrame,system_name::DataFrame) :: DataFrame  
  origin2system::Dict = Dict()
  
  #p = Progress(length(eachrow(system_name)),dt=0.5,barglyphs=BarGlyphs("[=> ]"),barlen=50)
  for ob in eachrow(system_name)
    origin2system[ob[1]] = ob[2]

  end
  ds = transform(dataSet,:name => ByRow(x -> get(origin2system,x,x)),renamecols=false)
  return ds
end


function geFeature(dataSet) :: DataFrame 
  # empty dataFrame
  key = unique(dataSet[:,:name])[:] :: Vector
  features = unique(dataSet[:,:terms][:]) :: Vector
  pushfirst!(features,"name")
  dataMatrix = hcat(key,zeros(length(key),length(features)-1)) :: Matrix
  dataFrame = DataFrame(dataMatrix,Symbol.(features))

  # initialize dataFrame
  for row in eachrow(dataSet)
    x = findfirst(x -> x==row[1],key) :: Int64
    y = row[2] :: String15
    dataFrame[x,y] = 1.0 :: Float64
  end
  
  return dataFrame
end

# output
#CSV.write("dataframe.csv",dataFrame)
#
# dataframe 只提供前行数 后列名的方式查找    如果想根据两个feature查找 必须先按值查找 将其中一个变为索引
# 在现代编程语言中--py,rust,julia,只有不可变/基础类型---数字,字符,元组--复制语义

function geAdj(dataFrame) :: DataFrame
  keys = dataFrame[!,:name][:] ::Vector{Any}     
  adjMatrix = hcat(keys,zeros(length(keys),length(keys))) :: Matrix
  keys = pushfirst!(keys,"name") 
  adjDf = DataFrame(adjMatrix,Symbol.(keys))
# metric : Tamimoto
  Tami(x,y) = (x⋅y) / (x⋅x + y⋅y - x⋅y)  
# initialize adjDf
  rows = eachrow(dataFrame)
  p = Progress(length(rows),dt=0.5,barglyphs=BarGlyphs("[=> ]"),barlen=50)
  Threads.@threads for i in eachindex(rows)
    rowp = rows[i]
    @inbounds for j in eachindex(rows)[i+1:end]
      rown = rows[j]
      sim = Tami(rowp[2:end],rown[2:end]) :: Float64
      idp = findfirst(x -> x==rowp[1],adjDf[!,:name]) :: Int64 
      idn = findfirst(x -> x==rown[1],adjDf[!,:name]) :: Int64
      adjDf[idp,rown[1]] = sim :: Float64
      adjDf[idn,rowp[1]] = sim :: Float64
    end
  next!(p)
  end
  return adjDf
end

#A = geAdj(dataFrame)
#CSV.write("../data/adj.csv",A)


# make degree dataframe
function geDegree(A) :: DataFrame
  keys = A[!,:name][:] :: Vector
  degreeMatrix = hcat(keys,zeros(length(keys),length(keys))) :: Matrix
  pushfirst!(keys,"name")
  degreeDf = DataFrame(degreeMatrix,Symbol.(keys))

  p = Progress(length(eachrow(A)),dt=0.5,barglyphs=BarGlyphs("[=> ]"),barlen=50)
  Threads.@threads for row in eachrow(A)
    degree = sum(row[2:end]) ::Float64
    id = findfirst(x -> x==row[1],degreeDf[!,:name]) :: Int64
    degreeDf[id,row[1]] = degree  :: Float64
    next!(p)
  end

  return degreeDf
end

#A = CSV.read("../data/adj.csv",DataFrame)
#D = geDegree(A) 
#CSV.write("../data/degree.csv",D)


#L = A .- D
#λ,U = eigen(L)
#O = L*U




