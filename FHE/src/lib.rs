use concrete::*;

fn sum_ct_vec(mut c: VectorLWE, new_min: f64) -> VectorLWE{
    let lenght = c.nb_ciphertexts;
    let mut ct_min = 0.;
    let mut min = 0.;
    let mut ct_min_arr = vec![0.; lenght];
    
    for i in 0..lenght{
        min = f64::abs(f64::min(0., c.encoders[i].get_min() as f64));
        ct_min += min;
        ct_min_arr[i] = min;
    }
    
    c.add_constant_static_encoder_inplace(&ct_min_arr).unwrap();
    let mut ct = c.sum_with_new_min(ct_min+new_min).unwrap();
    ct.add_constant_dynamic_encoder_inplace(&[-1.*ct_min]).unwrap();
    
    return ct;
}

fn sum_N(x: &VectorLWE) -> VectorLWE{
    let mut y = x.clone();
    let mut number = x.nb_ciphertexts as f64;
    let mut n = 0;
    while number/2. == f64::floor(number/2.){
        n += 1;
        number /= 2.;
    }
    let padd = x.encoders[0].nb_bit_padding;
    let mut ct_1: VectorLWE;
    let mut ct_2: VectorLWE;
    

    for i in 0..(n as usize){
        if ((padd as i32) - (n as i32) <= 0) && (y.encoders[0].nb_bit_padding == 1){
            println!("Not enough padding!");
            return y;
        }else{
            let N = u32::pow(2, (n-i-1) as u32) as usize;
            let mut tmpVec = VectorLWE::zero(x.dimension, N).unwrap();
            for j in 0..N{
                ct_1 = y.extract_nth(2*j).unwrap();
                ct_2 = y.extract_nth(2*j+1).unwrap();

                ct_1.add_with_padding_inplace(&ct_2).unwrap();

                tmpVec.copy_in_nth_nth_inplace(j, &ct_1, 0).unwrap();
            }
            y = tmpVec.clone();
        }
    }
    if y.nb_ciphertexts > 1{
        y = sum_ct_vec(y, 0.);
    }
    return y;    
}

fn relu(x: f64) -> f64{
    return f64::max(x, 0.);
}

fn elu_plus_one(x: f64) -> f64{
    if x >= 0. {
        return x+1. ;
    }
    else {
        return f64::exp(x)
    }
}

pub struct Net{
    Layer_1: Vec<Vec<f64>>,
    Layer_2: Vec<Vec<f64>>,
    Layer_mu: Vec<Vec<f64>>,
    Layer_sig: Vec<Vec<f64>>
}
impl Net{

    pub fn new() -> Self{
        Net{
            Layer_1: vec![vec![0.1; 19+1]; 12],
            Layer_2: vec![vec![0.1; 12+0]; 12],
            Layer_mu: vec![vec![0.1; 12+0]; 6],
            Layer_sig: vec![vec![0.1; 12+0]; 6]
        }
    }

    pub fn forward(&self, input: VectorLWE, bsk: &LWEBSK) -> (VectorLWE, VectorLWE){

        let bit_precision = 5;
        let enc_out = Encoder::new(0., 10., 5, 7).unwrap();

        let mut output_1 = VectorLWE::zero(input.dimension, 12).unwrap();
        let mut output_2 = VectorLWE::zero(input.dimension, 12).unwrap();
        let mut output_mu = VectorLWE::zero(input.dimension, 6).unwrap();
        let mut output_sig = VectorLWE::zero(input.dimension, 6).unwrap();

        for (i, weights) in self.Layer_1.iter().enumerate(){
            let mut ct_tmp = input.mul_constant_with_padding(weights, 1., bit_precision).unwrap();
            ct_tmp = sum_N(&ct_tmp);
            //add bias
            output_1.copy_in_nth_nth_inplace(i, &(ct_tmp.bootstrap_nth_with_function(&bsk, |x| relu(x), &enc_out, 0).unwrap()), 0).unwrap();
        }

        for (i, weights) in self.Layer_2.iter().enumerate(){
            let mut ct_tmp = output_1.mul_constant_with_padding(weights, 1., bit_precision).unwrap();
            ct_tmp = sum_N(&ct_tmp);
            ct_tmp.pp();
            //add bias
            output_2.copy_in_nth_nth_inplace(i, &(ct_tmp.bootstrap_nth_with_function(&bsk, |x| relu(x), &enc_out, 0).unwrap()), 0).unwrap();
        }

        for (i, weights) in self.Layer_mu.iter().enumerate(){
            let mut ct_tmp = output_2.mul_constant_with_padding(weights, 1., bit_precision).unwrap();
            ct_tmp = sum_N(&ct_tmp);
            ct_tmp.pp();
            //add bias
            output_mu.copy_in_nth_nth_inplace(i, &ct_tmp, 0).unwrap();
        }

        for (i, weights) in self.Layer_sig.iter().enumerate(){
            let mut ct_tmp = output_2.mul_constant_with_padding(weights, 1., bit_precision).unwrap();
            ct_tmp = sum_N(&ct_tmp);
            ct_tmp.pp();
            //add bias
            output_sig.copy_in_nth_nth_inplace(i, &(ct_tmp.bootstrap_nth_with_function(&bsk, |x| elu_plus_one(x), &enc_out, 0).unwrap()), 0).unwrap();
        }
        /*
        for (i, ctxt) in input.iter().enumerate() {
            //output_1[i] = 
        }
        */
        return (output_mu, output_sig);
    }
}
