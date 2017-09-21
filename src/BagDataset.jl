import Base: getindex;

export BagDataset, getindex;

type BagDataset{T, U} <: AbstractDataset
	X::AbstractMatrix{T};
	bags::AbstractVector{UnitRange{Int}};
	Y::AbstractVector{U};
end

function getindex(dataset::BagDataset, index::Int)
	return (sliceMatrix(dataset.X[dataset.bags[index]]), dataset.Y[index]);
end
