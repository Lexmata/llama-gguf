//! Verify RoPE frequency computation matches llama.cpp exactly.

fn main() {
    let head_dim = 64;
    let freq_base = 1000000.0f32;
    
    println!("=== RoPE Frequency Verification ===");
    println!("head_dim = {}, freq_base = {}", head_dim, freq_base);
    println!();
    
    // llama.cpp uses: theta = pow(freq_base, -2.0f*(float)i0/n_dims)
    // which is: 1.0 / pow(freq_base, 2*i/head_dim)
    // which is: pow(freq_base, -(2*i/head_dim))
    
    // Our formula: 1.0 / freq_base.powf((2 * i) as f32 / head_dim as f32)
    // which is: freq_base^(-(2*i/head_dim))
    
    // These should be equivalent.
    
    println!("Dimension | Our Freq    | llama.cpp style | Match?");
    println!("----------+-------------+-----------------+--------");
    
    for i in 0..8 {
        let our_freq = 1.0 / freq_base.powf((2 * i) as f32 / head_dim as f32);
        let llama_freq = freq_base.powf(-2.0 * i as f32 / head_dim as f32);
        let match_str = if (our_freq - llama_freq).abs() < 1e-10 { "✓" } else { "✗" };
        println!("{:9} | {:11.9} | {:15.9} | {}", i, our_freq, llama_freq, match_str);
    }
    
    println!();
    println!("=== RoPE Position Encoding ===");
    
    // Test position encoding for position 0 and 1
    for pos in 0..2 {
        println!("\nPosition {}:", pos);
        let position = pos as f32;
        
        for i in 0..4 {
            let freq = 1.0 / freq_base.powf((2 * i) as f32 / head_dim as f32);
            let theta = position * freq;
            let cos = theta.cos();
            let sin = theta.sin();
            println!("  dim {:2}: freq={:.9}, theta={:.6}, cos={:+.6}, sin={:+.6}", 
                2*i, freq, theta, cos, sin);
        }
    }
    
    println!();
    println!("=== Sample RoPE Application ===");
    
    // Apply RoPE to a sample vector at position 0 and 1
    let sample = vec![1.0f32, 0.5, -0.3, 0.8, 0.2, -0.1, 0.6, -0.4];
    
    for pos in 0..2 {
        let mut data = sample.clone();
        let position = pos as f32;
        
        for i in 0..4 {
            let freq = 1.0 / freq_base.powf((2 * i) as f32 / head_dim as f32);
            let theta = position * freq;
            let cos = theta.cos();
            let sin = theta.sin();
            
            let x0 = data[2*i];
            let x1 = data[2*i + 1];
            
            data[2*i] = x0 * cos - x1 * sin;
            data[2*i + 1] = x0 * sin + x1 * cos;
        }
        
        println!("Position {}: {:?}", pos, data);
    }
    
    // For position 0, RoPE should be identity (theta=0 for all dims)
    println!("\nNote: At position 0, theta=0 for all dimensions, so cos=1, sin=0.");
    println!("This means position 0 RoPE is identity: x' = x*1 - y*0 = x, y' = x*0 + y*1 = y");
}
