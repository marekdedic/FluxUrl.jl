import Requires

export UrlDataset;

const UrlDataset{T<:AbstractFloat, Ti <:Integer} = DoubleBagDataset{SparseMatrixCSC{T, Ti}, Int};

function UrlDataset(dataset::AbstractVector{Tuple{T, U}})::UrlDataset where {T <: AbstractVector, U}
	return MultiBagDataset(dataset; sparse = true);
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
			bags[i] = k:(k + dl + pl + ql - 1);
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
