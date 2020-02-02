use tch::nn::ModuleT;
use tch::Tensor;
use tch::vision::{ vgg, imagenet };

use std::path::{ Path, PathBuf };

#[derive(Debug)]
struct Layers {
    pub layers: std::vec::Vec<std::boxed::Box<(dyn tch::nn::ModuleT)>>,
}

impl ModuleT for Layers {
    fn forward_t(&self, xs: &Tensor, train: bool) -> Tensor {
        if self.layers.is_empty() {
            xs.shallow_clone()
        } else {
            let xs = self.layers[0].forward_t(xs, train);

            self.layers
                .iter()
                .skip(1)
                .fold(xs, |xs, layer| layer.forward_t(&xs, train))
        }
    }
}

fn main() -> failure::Fallible<()> {
    let mut vs = tch::nn::VarStore::new(tch::Device::Cpu);
    let net = vgg::vgg16(&vs.root(), imagenet::CLASS_COUNT);

    let mut layer_exposed: Layers = unsafe {
        std::mem::transmute(net)
    };
    layer_exposed.layers.pop();

    let weights = Path::new("models/vgg16.ot");
    vs.load(weights)?;

    let annoy = rannoy::Rannoy::new(4096);
    
    let ann = Path::new("./image.ann");
    if !ann.exists() {
        let mut image_count:i32 = 0;    

        for (i, entry) in glob::glob("data/flowers/tulip/*.jpg").expect("Failed to read glob pattern").enumerate() {
            match entry {
                Ok(path) => {
                    println!("{} {:?}", i, path);

                    let image = imagenet::load_image_and_resize224(path)?;
                    let output = layer_exposed
                        .forward_t(&image.unsqueeze(0), false)
                        .softmax(-1, tch::Kind::Float); 

                    let results = Vec::<f32>::from(output);
                    annoy.add_item(i as i32, &results);
                    
                    image_count += 1;
                },
                _ => {}
            }
        }

        annoy.build(image_count);
        annoy.save(PathBuf::from("./image.ann"));
    }

    annoy.load(PathBuf::from("./image.ann"));

    let test_image = imagenet::load_image_and_resize224(
        Path::new("./test.jpg")
    )?;
    let test_output = layer_exposed
        .forward_t(&test_image.unsqueeze(0), false)
        .softmax(-1, tch::Kind::Float); 

    let (result, _distance) = annoy.get_nns_by_vector(Vec::<f32>::from(test_output), 3, -1);

    dbg!(result);
    dbg!(_distance);

    Ok(())
}
