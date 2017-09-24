import Requires

export UrlDataset;

const UrlDataset{T<:AbstractFloat} = DoubleBagDataset{SparseMatrixCSC{T}, Vector{UnitRange{Int}}, Int};

function UrlDataset{T<:AbstractFloat}(features::AbstractMatrix{T}, labels::AbstractVector{Int}, urlIDs::AbstractVector{Int}, urlParts::AbstractVector{Int}; info::AbstractVector{AbstractString} = Vector{AbstractString}(0))::UrlDataset
	if(!issorted(urlIDs))
		permutation = sortperm(urlIDs);
		features = features[:, permutation];
		labels = labels[permutation];
		urlIDs = urlIDs[permutation];
		urlParts = urlParts[permutation];
	end
	bags = findranges(urlIDs);
	subbags = Vector{Vector{UnitRange{Int}}}(length(bags));
	bagLabels = map(b->maximum(labels[b]), bags);
	for (i, bag) in enumerate(bags)
		if(!issorted(urlParts[bag]))
			permutation = sortperm(urlParts[bag]);
			features[:, bag] = features[:, bag][:, permutation];
			urlParts[bag] = urlParts[bag][permutation];
		end
		subbags[i] = findranges(urlParts[bag]);
	end
	return DoubleBagDataset{SparseMatrixCSC{T}, Vector{UnitRange{Int}}, Int}(SparseMatrixCSC(features), bags, subbags, bagLabels);
end

Requires.@require Datasets begin

	function UrlDataset(ds::Datasets.UrlDataset)
		features = spzeros(Float32, size(ds.domain.data, 2), size(ds.domain.data, 1) + size(ds.path.data, 1) + size(ds.query.data, 1));
		bags = Vector{UnitRange{Int}}(length(ds.target));
		subbags = Vector{Vector{UnitRange{Int}}}(length(ds.target));
		k = 1;
		for i in 1:length(ds.target)
			dl = length(ds.domain.bags[i])
			pl = length(ds.path.bags[i])
			ql = length(ds.query.bags[i])
			bags[i] = k:(k + dl + pl + ql);
			subbags[i] = [1:dl,(dl + 1):(dl + pl),(dl + pl + 1):(dl + pl + ql)]
			for j in 1:dl
				features[:, k] = ds.domain.data[ds.domain.bags[i], :][j, :];
				k += 1;
			end
			for j in 1:pl
				features[:, k] = ds.path.data[ds.path.bags[i], :][j, :];
				k += 1;
			end
			for j in 1:ql
				features[:, k] = ds.query.data[ds.query.bags[i], :][j, :];
				k += 1;
			end
		end
		return DoubleBagDataset(features, bags, subbags, ds.target);
	end

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
