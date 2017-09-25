import Base: start, next, done;

export AbstractDataset, start, next, done, sample;

abstract type AbstractDataset end

function start(::AbstractDataset)::Int
	return 1;
end

function next(dataset::AbstractDataset, state::Int)::Tuple{Any, Int}
	return (dataset[state], state + 1);
end

function done(dataset::AbstractDataset, state::Int)::Bool
	return state > length(dataset.Y);
end

function sample(dataset::AbstractDataset, count::Int)
	indices = StatsBase.sample(1:length(dataset), count);
	return map(i->dataset[i], indices)
end

function sliceMatrix(X::AbstractMatrix)::AbstractVector{AbstractVector}
	return map(y->view(X, :, y), 1:size(X, 2));
end

function sliceMatrix(X::AbstractMatrix, ranges::AbstractVector{UnitRange{Int}})::AbstractVector{AbstractArray}
	return map(y->view(X, :, y), ranges);
end

sliceMatrix(X::AbstractVector) = [X];
sliceMatrix(X::AbstractVector, ranges::AbstractVector{UnitRange{Int}}) = [X];
