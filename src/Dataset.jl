import Base: getindex, length;

export Dataset, getindex, length;

struct Dataset{T, U} <: AbstractDataset
	X::AbstractMatrix{T};
	Y::AbstractVector{U};
end

function getindex(dataset::Dataset, index::Int)
	return (view(dataset.X, :, index), dataset.Y[index]);
end

function length(dataset::Dataset)::Int
	return size(dataset.X, 2);
end
