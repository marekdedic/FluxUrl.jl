import Base: getindex;

export DoubleBagDataset, getindex;

type DoubleBagDataset{T, U} <: AbstractDataset
	X::AbstractMatrix{T};
	bags::AbstractVector{UnitRange{Int}};
	subbags::AbstractVector{UnitRange{Int}};
	Y::AbstractVector{U};
end

function getindex(dataset::DoubleBagDataset, index::Int)
	return (map(sliceMatrix, sliceMatrix(dataset.X[dataset.bags[index]], dataset.subbags[dataset.bags[index]])), dataset.Y[index]);
end
