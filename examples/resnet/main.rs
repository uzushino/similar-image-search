use std::mem;
use tch::nn::ModuleT;
use tch::Tensor;
use tch::vision::{ resnet, imagenet };
use tch::{nn, nn::Conv2D, nn::FuncT, nn::SequentialT, nn::seq_t};

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

/*
fn downsample(p: nn::Path, c_in: i64, c_out: i64, stride: i64) -> SequentialT {
    if stride != 1 || c_in != c_out {
        nn::seq_t()
            .add(conv2d(&p / "0", c_in, c_out, 1, 0, stride))
            .add(nn::batch_norm2d(&p / "1", c_out, Default::default()))
    } else {
        nn::seq_t()
    }
}


fn basic_block(p: nn::Path, c_in: i64, c_out: i64, stride: i64) -> SequentialT {
    let conv1 = conv2d(&p / "conv1", c_in, c_out, 3, 1, stride);
    let bn1 = nn::batch_norm2d(&p / "bn1", c_out, Default::default());
    let conv2 = conv2d(&p / "conv2", c_out, c_out, 3, 1, 1);
    let bn2 = nn::batch_norm2d(&p / "bn2", c_out, Default::default());
    let downsample = downsample(&p / "downsample", c_in, c_out, stride);
    nn::func_t(move |xs, train| {
        let ys = xs
            .apply(&conv1)
            .apply_t(&bn1, train)
            .relu()
            .apply(&conv2)
            .apply_t(&bn2, train);
        (xs.apply_t(&downsample, train) + ys).relu()
    })
}

fn basic_layer(p: nn::Path, c_in: i64, c_out: i64, stride: i64, cnt: i64) -> SequentialT {
    let mut layer = nn::seq_t().add(basic_block(&p / "0", c_in, c_out, stride));
    for block_index in 1..cnt {
        layer = layer.add(basic_block(&p / &block_index.to_string(), c_out, c_out, 1))
    }
    layer
}

fn conv2d(p: nn::Path, c_in: i64, c_out: i64, ksize: i64, padding: i64, stride: i64) -> Conv2D {
    let conv2d_cfg = nn::ConvConfig {
        stride,
        padding,
        bias: false,
        ..Default::default()
    };
    nn::conv2d(&p, c_in, c_out, ksize, conv2d_cfg)
}

fn resnet(
    p: &nn::Path,
    nclasses: Option<i64>,
    c1: i64,
    c2: i64,
    c3: i64,
    c4: i64,
) -> SequentialT {
    let conv1 = conv2d(p / "conv1", 3, 64, 7, 3, 2);
    let bn1 = nn::batch_norm2d(p / "bn1", 64, Default::default());
    let layer1 = basic_layer(p / "layer1", 64, 64, 1, c1);
    let layer2 = basic_layer(p / "layer2", 64, 128, 2, c2);
    let layer3 = basic_layer(p / "layer3", 128, 256, 2, c3);
    let layer4 = basic_layer(p / "layer4", 256, 512, 2, c4);
    let fc = nclasses.map(|n| nn::linear(p / "fc", 512, n, Default::default()));
    let seq = nn::seq_t();

    seq_t.apply(&conv1)
        .apply_t(&bn1, train)
        .relu()
        .max_pool2d(&[3, 3], &[2, 2], &[1, 1], &[1, 1], false)
        .apply_t(&layer1, train)
        .apply_t(&layer2, train)
        .apply_t(&layer3, train)
        .apply_t(&layer4, train)
        .adaptive_avg_pool2d(&[1, 1])
        .flat_view()
        .apply_opt(&fc)
}

pub fn resnet18(p: &nn::Path, num_classes: i64) -> impl ModuleT {
    resnet(p, Some(num_classes), 2, 2, 2, 2)
}
*/

fn main() -> failure::Fallible<()> {
    let mut vs = tch::nn::VarStore::new(tch::Device::Cpu);
    //let net = vgg::vgg16(&vs.root(), imagenet::CLASS_COUNT);
    let net = resnet::resnet18_no_final_layer(&vs.root());
    //let net = resnet::resnet18(&vs.root(), imagenet::CLASS_COUNT);
    //let net = resnet18(&vs.root(), imagenet::CLASS_COUNT);
    /*
    let mut layer_exposed: Layers = unsafe {
        mem::transmute(net)
    };
    layer_exposed.layers.pop();
    */
    let weights = Path::new("/tmp/resnet18.pth");
    vs.load(weights)?;

    let annoy = rannoy::Rannoy::new(4096);

    let ann = Path::new("resnet.ann");
    if !ann.exists() {
        let mut image_count:i32 = 0;

        for (i, entry) in glob::glob("./flowers/tulip/*.jpg").expect("Failed to read glob pattern").enumerate() {
            match entry {
                Ok(path) => {
                    let image = imagenet::load_image_and_resize224(path.clone())?;
                    
                    let output = net 
                        .forward_t(&image.unsqueeze(0), false)
                        .softmax(-1, tch::Kind::Float); 

                    dbg!(&output);
                    panic!();

                    let results = Vec::<f32>::from(output);
                    annoy.add_item(i as i32, &results);

                    println!("{:?} : {:?}", image_count, path.clone());

                    image_count += 1;
                },
                _ => {}
            }
        }

        annoy.build(image_count);
        annoy.save(PathBuf::from("resnet.ann"));
    }

    annoy.load(PathBuf::from("resnet.ann"));
    
    let test_image = imagenet::load_image_and_resize224(
        Path::new("./test.jpg")
    )?;
    let test_output = net
        .forward_t(&test_image.unsqueeze(0), false)
        .softmax(-1, tch::Kind::Float); 

    let (result, _distance) = annoy.get_nns_by_vector(Vec::<f32>::from(test_output), 3, -1);

    dbg!(result);
    dbg!(_distance);

    Ok(())
}
