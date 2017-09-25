import Base: getindex, length;

export MultiBagDataset, getindex, length;

struct MultiBagDataset{T <: AbstractMatrix, U <: Union{AbstractVector, UnitRange{Int}}, V} <: AbstractDataset
	X::T;
	bags::AbstractVector{U};
	Y::AbstractVector{V};
end

function getindex(dataset::MultiBagDataset, index::Int)::Tuple
	return (_getindex_populate(dataset.X, dataset.bags[index]), dataset.Y[index]);
end

function _getindex_populate(X::T, bag::AbstractVector)::AbstractVector where T <: AbstractMatrix
	if typeof(bag) <: UnitRange
		return sliceMatrix(view(X, :, bag));
	else
		return map(y->_getindex_populate(X, y) , bag);
	end
end

function length(dataset::MultiBagDataset)::Int
	return length(dataset.bags);
end

const SingleBagDataset{T <: AbstractMatrix, U} = MultiBagDataset{T, UnitRange{Int}, U};
const DoubleBagDataset{T <: AbstractMatrix, U} = MultiBagDataset{T, AbstractVector{UnitRange{Int}}, U};
const TripleBagDataset{T <: AbstractMatrix, U} = MultiBagDataset{T, AbstractVector{AbstractVector{UnitRange{Int}}}, U}; # TODO: CHECK validity
