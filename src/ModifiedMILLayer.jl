import Flux;

export ModifiedMILLayer;

struct ModifiedMILLayer <: Flux.AbstractLayer
	NN::AbstractVector{Flux.AbstractLayer};
	aggregation::Function;
end

function ModifiedMILLayer(NN::Flux.AbstractLayer, instanceCount::Int, aggregation::Function)::ModifiedMILLayer
	NNvec = Vector{Flux.AbstractLayer}(instanceCount);
	NNvec[1] = NN;
	NNvec[2:end] .= deepcopy(NN);
	return ModifiedMILLayer(NNvec, aggregation);
end

(a::ModifiedMILLayer)(x) = a.aggregation(map(i->(a.NN[i])(x[i]), 1:length(x)));
