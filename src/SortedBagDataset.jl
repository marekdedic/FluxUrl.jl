import Base: size, vcat, length;
import Base.Operators.getindex;
import StatsBase: sample, nobs;
import DataFrames;

export SortedBagDataset

"""In this dataset sub-bags is an array of arrays. Each array within an array defines indexes in X t, where each element of the master array corresponds to one subBag, 
and the array within holds indexes of instances belonging to the subBag.
 """
type SortedBagDataset{T<:AbstractFloat}
  x::AbstractArray{T,2};
  y::AbstractArray{Int,1};
  bags::Array{UnitRange{Int64},1};

  info::DataFrames.DataFrame;
end

function SortedBagDataset{T<:AbstractFloat}(x::AbstractArray{T,2},y::AbstractArray{Int,1},bags::Array{UnitRange{Int64},1};info::DataFrames.DataFrame=DataFrames.DataFrame([]))
  return(SortedBagDataset(x,y,bags,info))
end


function SortedBagDataset{T<:AbstractFloat}(x::AbstractArray{T,2},y::AbstractArray{Int,1},bagids::AbstractArray;info::DataFrames.DataFrame=DataFrames.DataFrame([]))
  I=sortperm(bagids)
  bagids=bagids[I];
  x=x[:,I]
  y=y[I]
  if !isempty(info)
    info=info[I,:]
  end

  #create the bags and subbags
  bags=findranges(bagids);
  bagy=map(bag->maximum(y[bag]),bags);
  return(SortedBagDataset(x,bagy,bags,info))
end

function nobs(ds::SortedBagDataset)
  length(ds.y)
end

function getindex(ds::SortedBagDataset,i::Int)
  getindex(ds::SortedBagDataset,[i])
end

function getindex(ds::SortedBagDataset,bagindexes::AbstractArray{Int})
  (bags,instanceidxs)=remap(ds.bags[bagindexes]);
  return(SortedBagDataset(ds.x[:,instanceidxs],ds.y[bagindexes],bags,ds.info[instanceidxs,:]))
end

function sample(ds::SortedBagDataset,n::Int64)
  indexes=sample(1:length(ds.bags),min(n,length(ds.y)),replace=false);
  return(getindex(ds,indexes));
end

function sample(ds::SortedBagDataset,n::Array{Int64})
  classbagids=map(i->findn(ds.y.==i),1:maximum(ds.y));
  indexes=mapreduce(i->sample(classbagids[i],min(length(classbagids[i]),n[i]);replace=true),append!,1:length(n));
  return(getindex(ds,indexes));
end

function vcat(d1::SortedBagDataset,d2::SortedBagDataset)
  #we need to redefine bags and sub-bags, of them needs to be shifted by the number of bags / instances in d1
  l=size(d1.x,2);
  bags=vcat(deepcopy(d1.bags),map(i->i+l,d2.bags));
  l=size(d1.x,2)
  if !isempty(d1.info) && !isempty(d2.info)
    ds=SortedBagDataset(hcat(d1.x,d2.x),vcat(d1.y,d2.y),bags,vcat(d1.info,d2.info));
  else
    ds=SortedBagDataset(hcat(d1.x,d2.x),vcat(d1.y,d2.y),bags,DataFrames.DataFrame());
  end
  return(ds)
end


function vcat(dss::Array{SortedBagDataset})
  #we need to redefine bags and sub-bags, of them needs to be shifted by the number of bags / instances in d1
  ni=mapreduce(ds->size(ds.x,2),+,dss)
  nb=mapreduce(ds->length(ds.y),+,dss)
  fMat=zeros(size(ds.x,1),ni);

  i=1;
  for ds in enumerate(dss)
    fMat[:,i:i+size(ds.x,2)-1]=ds.x;
    i+=size(ds.x,2)
  end

  l=size(d1.x,2);
  bags=vcat(deepcopy(d1.bags),map(i->i+l,d2.bags));
  l=size(d1.x,2)
  if !isempty(d1.info) && !isempty(d2.info)
    ds=SortedBagDataset(hcat(d1.x,d2.x),vcat(d1.y,d2.y),bags,vcat(d1.info,d2.info));
  else
    ds=SortedBagDataset(hcat(d1.x,d2.x),vcat(d1.y,d2.y),bags,DataFrames.DataFrame());
  end
  return(ds)
end

function fgradient!(layers::Tuple,loss,ds::SortedBagDataset,glayers)
  fgradient!(layers::Tuple,loss,ds.x,(ds.bags,),ds.y,glayers)
end

function project!(layers::Tuple,ds::SortedBagDataset)
  project!(layers::Tuple,ds.x,(ds.bags,))
end

function forward!(layers::Tuple,ds::SortedBagDataset)
  forward!(layers::Tuple,ds.x,(ds.bags,))
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
