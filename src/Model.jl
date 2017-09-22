import Flux;

export UrlModel;

UrlModel(urlPartNN::Flux.Chain, urlPartAggregation::Function, urlNN::Flux.Chain)::Flux.AbstractLayer = Flux.Chain(ModifiedMILLayer(MILLayer(urlPartNN, urlPartAggregation), 3, concatAggregation), urlNN);
