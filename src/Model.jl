import Flux;

export UrlModel;

UrlModel(urlPartNN::Flux.Chain, urlPartAggregation::Function, urlNN::Flux.Chain) = Flux.Chain(ModifiedMILLayer(MILLayer(urlPartNN, urlPartAggregation), 3, concatAggregation), urlNN);
