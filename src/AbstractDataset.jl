import Base: start, next, done;

export AbstractDataset, start, next, done;

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

function sliceMatrix(X::AbstractMatrix)::AbstractVector{AbstractVector}
	return map(y->view(X, :, y), 1:size(X, 2));
end

function sliceMatrix(X::AbstractMatrix, ranges::AbstractVector{UnitRange})::AbstractVector{AbstractArray}
	return map(y->view(X, :, y), ranges);
end
