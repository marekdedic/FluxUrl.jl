import Base: getindex;

export Dataset, getindex;

struct Dataset{T, U} <: AbstractDataset
	X::AbstractMatrix{T};
	Y::AbstractVector{U};
end

function getindex(dataset::Dataset, index::Int)
	return (view(dataset.X, :, index), dataset.Y[index]);
end
