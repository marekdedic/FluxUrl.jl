import Flux;

export MILLayer;

struct MILLayer <: Flux.AbstractLayer
	NN::Flux.AbstractLayer;
	aggregation::Function;
end

(a::MILLayer)(x) = a.aggregation(map(y->a.NN(y), x));
