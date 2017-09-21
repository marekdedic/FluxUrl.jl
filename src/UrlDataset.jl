export UrlDataset;

const UrlDataset{T<:AbstractFloat} = DoubleBagDataset{T, Int};

function UrlDataset{T<:AbstractFloat}(features::Matrix{T}, labels::Vector{Int}, urlIDs::Vector{Int}, urlParts::Vector{Int})::UrlDataset
	if(!issorted(urlIDs))
		permutation = sortperm(urlIDs);
		features = features[:, permutation];
		labels = labels[permutation];
		urlIDs = urlIDs[permutation];
		urlParts = urlParts[permutation];
	end
	bags = findranges(urlIDs);
	subbags = Vector{UnitRange{Int}}(length(bags));
	bagLabels = map(b->maximum(labels[b]), bags);
	for i, bag in enumerate(bags)
		bagfeatures = view(features, :, bag);
		if(!issorted(urlParts[bag]))
			permutation = sortperm(urlParts[bag]);
			bagfeatures = bagfeatures[:, permutation];
			urlParts[bag] = urlParts[bag][permutation];
		end
		subbags[i] = findranges(urlParts[bag]);
	end
	return UrlDataset(features, bags, subbags, labels);
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
