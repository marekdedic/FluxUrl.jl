import Base: getindex, length;

export DoubleBagDataset, getindex, length;

struct DoubleBagDataset{T, U} <: AbstractDataset
	X::AbstractMatrix{T};
	bags::AbstractVector{UnitRange{Int}};
	subbags::AbstractVector{UnitRange{Int}};
	Y::AbstractVector{U};
end

function getindex(dataset::DoubleBagDataset, index::Int)
	return (map(sliceMatrix, sliceMatrix(dataset.X[dataset.bags[index]], dataset.subbags[dataset.bags[index]])), dataset.Y[index]);
end

function length(dataset::DoubleBagDataset)::Int
	return length(dataset.bags);
end
