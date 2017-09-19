import Base: Operators.getindex, vcat;
import DataFrames;

export UrlDataset, getindex, vcat, sample;

type UrlDataset{T<:AbstractFloat}
	domains::SortedBagDataset{T}
	paths::SortedBagDataset{T}
	queries::SortedBagDataset{T}

	y::Vector{Int}
	info::DataFrames.DataFrame;
end

function UrlDataset{T<:AbstractFloat}(features::Matrix{T}, labels::Vector{Int}, urlIDs::Vector{Int}, urlParts::Vector{Int}; info::Vector{AbstractString} = Vector{AbstractString}(0))::UrlDataset
	if(!issorted(urlIDs))
		permutation = sortperm(urlIDs);
		features = features[:, permutation];
		labels = labels[permutation];
		urlIDs = urlIDs[permutation];
		urlParts = urlParts[permutation];
		if size(info, 1) != 0;
			info = info[permutation];
		end
	end

	(domainFeatures, pathFeatures, queryFeatures) = map(i->features[:, urlParts .== i], 1:3);
	(domainBags, pathBags, queryBags) = map(i->findranges(urlIDs[urlParts .== i]), 1:3);

	subbags = findranges(urlIDs);
	bagLabels = map(b->maximum(labels[b]), subbags);
	if size(info, 1) != 0;
		bagInfo = map(b->info[b][1], subbags);
	else
		bagInfo = Vector{AbstractString}(0);
	end

	domains = SortedBagDataset(domainFeatures, bagLabels, domainBags);
	paths = SortedBagDataset(pathFeatures, bagLabels, pathBags);
	queries = SortedBagDataset(queryFeatures, bagLabels, queryBags);
	UrlDataset(domains, paths, queries, bagLabels, DataFrames.DataFrame(url = bagInfo))
end

function getindex(dataset::UrlDataset, i::Int)::UrlDataset
	return getindex(dataset, [i]);
end

function getindex(dataset::UrlDataset, indices::Vector{Int})::UrlDataset
	if size(dataset.info, 1) == 0
		info = DataFrames.DataFrame(url = Vector{AbstractString}(0));
	else
		info = dataset.info[indices, :];
	end
	UrlDataset(dataset.domains[indices], dataset.paths[indices], dataset.queries[indices], dataset.y[indices], info)
end

function vcat(d1::UrlDataset,d2::UrlDataset)
	UrlDataset(vcat(d1.domains,d2.domains), vcat(d1.paths,d2.paths), vcat(d1.queries,d2.queries), vcat(d1.y,d2.y), vcat(d1.info, d2.info))
end

function sample(dataset::UrlDataset, n::Int64)
	indices = sample(1:length(dataset.y), min(n, length(dataset.y)), replace=false);
	return getindex(dataset, indices);
end

function sample(dataset::UrlDataset, n::Vector{Int})
  classbagids = map(i->findn(dataset.y .==i ), 1:maximum(dataset.y));
  indices = mapreduce(i->sample(classbagids[i], minimum([length(classbagids[i]), n[i]]); replace=false), append!, 1:min(length(classbagids), length(n)));
  return(getindex(dataset, indices));
end
