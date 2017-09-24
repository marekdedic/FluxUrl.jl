import Base: getindex, length;

export DoubleBagDataset, getindex, length;

struct DoubleBagDataset{T <: AbstractMatrix, U <: AbstractVector{UnitRange{Int}}, V} <: AbstractDataset
	X::T;
	bags::AbstractVector{UnitRange{Int}};
	subbags::AbstractVector{U};
	Y::AbstractVector{V};
end

function getindex(dataset::DoubleBagDataset, index::Int)
	return (map(sliceMatrix, sliceMatrix(dataset.X[:, dataset.bags[index]], dataset.subbags[index])), dataset.Y[index]);
end

function length(dataset::DoubleBagDataset)::Int
	return length(dataset.bags);
end
