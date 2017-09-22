import Base: getindex, length;

export BagDataset, getindex, length;

struct BagDataset{T, U} <: AbstractDataset
	X::AbstractMatrix{T};
	bags::AbstractVector{UnitRange{Int}};
	Y::AbstractVector{U};
end

function getindex(dataset::BagDataset, index::Int)
	return (sliceMatrix(dataset.X[:, dataset.bags[index]]), dataset.Y[index]);
end

function length(dataset::BagDataset)::Int
	return length(dataset.bags);
end
