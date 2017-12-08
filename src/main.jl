using Flux;
using FluxUrl;
using ProfileView;
using ThreadedMap;

model = UrlModel(Chain(ReLU(2053, 20)), meanPooling, Chain(ReLU(3*20, 3*20), Flux.softmax));

loss(x, y) = Flux.crossentropy(model(x), [y]);

for (i, ds) in enumerate(loadRandom(2000;batchSize = 1000, parallel = true));
	if i == 1
		Flux.train!(loss, ds, Flux.ADAM(params(model)));
	else
		Profile.clear();
		@profile Flux.train!(loss, ds, Flux.ADAM(params(model)));
		break;
	end
end

ProfileView.view();
