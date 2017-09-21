import Flux;

export ModifiedMILLayer;

struct ModifiedMILLayer
	NN::AbstractVector{Flux.Chain};
	aggregation::Function;
end

function ModifiedMILLayer(NN::Flux.Chain, instanceCount::Int, aggregation::Function)
	NNvec = Vector{Flux.Chain}(instanceCount);
	NNvec[1] = NN;
	NNvec[2:end] .= deepcopy(NN);
	return ModifiedMILLayer(NNvec, aggregation);
end

(a::ModifiedMILLayer)(x) = a.aggregation(map(y->a.NN(y), x));
