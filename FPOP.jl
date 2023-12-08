using Random
using Intervals
import Base.min

# Dealing with quadratic loss function \gamma(y_t, \mu) = (y_t - \mu)^2
mutable struct Quadratic
	a::Float64
	b::Float64
	c::Float64
	tau::Int64
	set::IntervalSet
end

function min(q::Quadratic)
	return -(q.b^2)/(4*q.a) + q.c
end

function roots(q::Quadratic, z::Float64)
	disc = q.b^2 - 4*q.a*(q.c -z)
	if disc < 0
		return IntervalSet()
	end
	x1 = (-q.b + sqrt(disc))/(2*q.a)
	x2 = (-q.b - sqrt(disc))/(2*q.a)
	return IntervalSet(min(x1,x2)..max(x1,x2))
end

function get_changepoints(chpts::Vector{Int64})
	CP = Vector{Int64}(undef, 0)
	n = length(chpts)
	last = chpts[n]
	push!(CP, last)
	while last > 0
		last = chpts[last]
		push!(CP, last)
	end
	sort!(CP)
	return CP
end

## sim data
Random.seed!(1234)
X = vcat(randn(1000), randn(1000) .+ 5, randn(1000) .+ 2)

## Set up initial vars for algorithm
n = length(X)
# penalty
beta = 2 * log(n)
# interval for mu
D = IntervalSet(minimum(X)..maximum(X))

cost_list = Vector{Quadratic}(undef, 0)
push!(cost_list, Quadratic(0, 0, 0, 0, D))
F = Vector{Float64}(undef, n)
chpts = Vector{Int64}(undef, n)

for t = 1:n
	# update costs with observation
	Nquads = length(cost_list)
	q_mins = Vector{Float64}(undef, Nquads)
	for i = 1:Nquads
		q = cost_list[i]
		q.a += 1
		q.b -= 2*X[t]
		q.c += X[t]^2
		q_mins[i] = min(q)
	end

	F[t], idx = findmin(q_mins)
	chpts[t] = cost_list[idx].tau

	# update sets
	I_union = Vector{IntervalSet}(undef, Nquads)
	quads_todelete = Vector{Int64}(undef, 0)
	for i = 1:Nquads
		q = cost_list[i]
		# I_t^tau
		It_tau = roots(q, F[t] + beta)
		# then update Set_t^tau = Set_t-1^tau n I_t^tau
		q.set = intersect(q.set, It_tau)
		# Store I intervals to do Set_t^t
		if i == 1
			I_union[i] = It_tau
		else
			I_union[i] = union(I_union[i-1], It_tau)
		end
		# Can we prune this quadratic
		if isempty(q.set)
			push!(quads_todelete, i)
		end
	end
	deleteat!(cost_list, quads_todelete)

	# Check if t can be pruned right away
	set_tt = setdiff(D, I_union[Nquads])
	if !isempty(set_tt)
		push!(cost_list, Quadratic(0, 0, F[t] + beta, t, set_tt))
	end
end

println(get_changepoints(chpts))
