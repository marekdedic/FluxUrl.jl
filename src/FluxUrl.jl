module FluxUrl

include("AbstractDataset.jl");
include("Dataset.jl");
include("BagDataset.jl");
include("DoubleBagDataset.jl");

include("layers.jl");
include("MILLayer.jl");
include("ModifiedMILLayer.jl");

include("UrlDataset.jl");

include("IterableParser.jl");

include("featureGenerators.jl");
include("labellers.jl");

include("loadDataset.jl");
include("processDataset.jl");

include("Model.jl");

end
