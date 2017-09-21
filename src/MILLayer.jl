import Flux;

export MILLayer;

struct MILLayer
	NN::Flux.Chain;
	aggregation::Function;
end

(a::MILLayer)(x) = a.aggregation(map(y->a.NN(y), x));
