import Flux;

export ModifiedMILLayer;

struct ModifiedMILLayer
	NN::AbstractVector;
	aggregation::Function;
end

function ModifiedMILLayer(NN, instanceCount::Int, aggregation::Function)::ModifiedMILLayer
	NNvec = Vector(instanceCount);
	NNvec[1] = NN;
	NNvec[2:end] .= deepcopy(NN);
	return ModifiedMILLayer(NNvec, aggregation);
end

(a::ModifiedMILLayer)(x::AbstractVector)::AbstractVector = a.aggregation(map(i->(a.NN[i])(x[i]), 1:length(x)));
