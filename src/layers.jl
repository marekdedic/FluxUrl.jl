import Flux;

export ReLU, Linear, concatAggregation, meanPooling;

f32init(dims...) = randn(Float32, dims...)/100;

ReLU(in::Int, out::Int) = Flux.Dense(in, out, Flux.relu; init = f32init);
Linear(in::Int, out::Int) = Flux.Dense(in, out, identity; init = f32init);

concatAggregation{T <: AbstractVector}(vecs::AbstractVector{T})::AbstractVector = vcat(vecs...);
meanPooling{T <: AbstractVector}(vecs::AbstractVector{T})::AbstractVector = begin
	result = vecs[1];
	for x in vecs[2:end]
		result += x;
	end
	result /= length(vecs);
end
