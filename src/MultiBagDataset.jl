import Base: getindex, length;

export MultiBagDataset, getindex, length;

struct MultiBagDataset{T <: AbstractMatrix, U <: Union{AbstractVector, UnitRange{Int}}, V} <: AbstractDataset
	X::T;
	bags::AbstractVector{U};
	Y::AbstractVector{V};
end

function MultiBagDataset(dataset::AbstractVector{Tuple{T, U}}; sparse::Bool = false)::MultiBagDataset where {T <: AbstractVector, U}
	function instanceType(instance::AbstractVector)::Type
		if eltype(instance) <: AbstractVector
			return instanceType(instance[1]);
		end
		return eltype(instance);
	end

	function instanceLength(instance::AbstractVector)::Int
		if eltype(instance) <: AbstractVector
			return instanceLength(instance[1]);
		end
		return length(instance);
	end

	function instanceCount(instance::AbstractVector)::Int
		if eltype(eltype(instance)) <: AbstractVector
			return mapreduce(x->instanceCount(x), +, instance);
		end
		return length(instance);
	end

	function insert(instance::AbstractVector, index::Int)::Int
		if eltype(instance) <: AbstractVector
			foreach(y->begin index = insert(y, index) end, instance);
			return index;
		else
			X[:, index] = instance;
			return index + 1;
		end
	end

	function bagify(instance::AbstractVector, index::Int)::Tuple{AbstractVector, Int}
		if eltype(eltype(instance)) <: AbstractVector
			bags = map(x->begin (y, index) = bagify(x, index); y end, instance)
			return (bags, index);
		else
			return (index:(index + length(instance) - 1), index + length(instance));
		end
	end

	X = Matrix{instanceType(dataset[1][1])}(instanceLength(dataset[1][1]), mapreduce(x->instanceCount(x[1]), +, dataset));
	if sparse
		X = Base.sparse(X);
	end

	index = 1;
	foreach(y->begin index = insert(y[1], index) end, dataset);
	index = 1;
	bags = map(x->begin (y, index) = bagify(x[1], index); y end, dataset);
	Y = map(x->x[2], dataset);
	return MultiBagDataset(X, bags, Y);
end

function getindex(dataset::MultiBagDataset, index::Int)::Tuple
	function slice(X::T, bag::AbstractVector)::AbstractVector where T <: AbstractMatrix
		if typeof(bag) <: UnitRange
			return sliceMatrix(view(X, :, bag));
		end
		return map(y->slice(X, y) , bag);
	end

	return (slice(dataset.X, dataset.bags[index]), dataset.Y[index]);
end

function length(dataset::MultiBagDataset)::Int
	return length(dataset.bags);
end

const SingleBagDataset{T <: AbstractMatrix, U} = MultiBagDataset{T, UnitRange{Int}, U};
const DoubleBagDataset{T <: AbstractMatrix, U} = MultiBagDataset{T, Vector{UnitRange{Int}}, U};
const TripleBagDataset{T <: AbstractMatrix, U} = MultiBagDataset{T, Vector{Vector{UnitRange{Int}}}, U};
