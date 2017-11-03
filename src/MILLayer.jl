import Flux;

export MILLayer;

struct MILLayer
	NN;
	aggregation::Function;
end

(a::MILLayer)(x::AbstractVector)::AbstractVector = a.aggregation(map(y->(a.NN)(y), x));
