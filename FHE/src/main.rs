use concrete::*;

mod lib;
use lib::Net;

fn main() {

    // #### ---- FHE STUFF ---- ####

    let lwe_dim = 1024;//512, 1024, 2048];
    let lwe_noise = -40;//-19, -40, -62];
    
    let rlwe_dim = 1024; //512, 1024, 2048];
    let rlwe_noise = -40; //-19, -40, -62];

    let base_log = 7;
    let lvl = 7;
    
    //let lwe_params: LWEParams = LWEParams::new(lwe_dim, lwe_noise);
    let rlwe_params: RLWEParams = RLWEParams{polynomial_size: rlwe_dim, dimension: 1, log2_std_dev: rlwe_noise};

    let sk_rlwe = RLWESecretKey::new(&rlwe_params);
    let sk = sk_rlwe.to_lwe_secret_key();
    let bsk = LWEBSK::new(&sk, &sk_rlwe, base_log, lvl);

    let data = vec![0.1; 19+1];
    let enc = Encoder::new(0., 1., 5, 7).unwrap();

    // #### ---- FHE STUFF ---- ####

    println!("done");

    let net = Net::new();

    let input = VectorLWE::encode_encrypt(&sk, &data, &enc).unwrap();

    let (mu, sig) = net.forward(input, &bsk);

    println!("mu = {:?}, sig = {:?}", mu.decrypt_decode(&sk).unwrap(), sig.decrypt_decode(&sk).unwrap());
    
}
