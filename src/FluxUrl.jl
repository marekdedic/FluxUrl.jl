module FluxUrl

include("SortedBagDataset.jl");
include("UrlDataset.jl");

include("IterableParser.jl");

include("featureGenerators.jl");
include("labellers.jl");

include("loadDataset.jl");
include("processDataset.jl");

end
