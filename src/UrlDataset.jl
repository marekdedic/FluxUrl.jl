export UrlDataset;

const UrlDataset{T<:AbstractFloat} = DoubleBagDataset{T, Int};

function UrlDataset{T<:AbstractFloat}(features::Matrix{T}, labels::Vector{Int}, urlIDs::Vector{Int}, urlParts::Vector{Int}; info::Vector{AbstractString} = Vector{AbstractString}(0))::UrlDataset
	if(!issorted(urlIDs))
		permutation = sortperm(urlIDs);
		features = features[:, permutation];
		labels = labels[permutation];
		urlIDs = urlIDs[permutation];
		urlParts = urlParts[permutation];
	end
	bags = findranges(urlIDs);
	subbags = Vector{UnitRange{Int}}(3*length(bags));
	bagLabels = map(b->maximum(labels[b]), bags);
	for bag in bags
		if(!issorted(urlParts[bag]))
			permutation = sortperm(urlParts[bag]);
			features[:, bag] = features[:, bag][:, permutation];
			urlParts[bag] = urlParts[bag][permutation];
		end
		subbags[bag] = findranges(urlParts[bag]);
	end
	return DoubleBagDataset(features, bags, subbags, bagLabels);
end

function findranges(ids::AbstractArray)
	if !issorted(ids)
		error("ids parameter should be sorted")
	end
	bags=fill(0:0,length(unique(ids)))
	idx=1
	bidx=1
	for i in 2:length(ids)
		if ids[i]!=ids[idx]
			bags[bidx]=idx:i-1
			idx=i;
			bidx+=1;
		end
	end
	if bidx<=length(bags)
		bags[bidx]=idx:length(ids)
	end
	return(bags)
end
