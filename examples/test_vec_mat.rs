// Test vec_mat computation manually
use llama_rs::tensor::{Tensor, DType};
use llama_rs::backend::cpu::CpuBackend;
use llama_rs::Backend;

fn main() {
    let backend = CpuBackend::new();
    
    // Create a simple test case
    // x = [1, 2, 3] (in_features = 3)
    // W = [[1, 4],   (shape [3, 2] meaning in=3, out=2)
    //      [2, 5],
    //      [3, 6]]
    // Expected: x @ W = [1*1+2*2+3*3, 1*4+2*5+3*6] = [14, 32]
    
    // In GGUF column-major format, W[i,j] = data[i + j * 3]
    // Column 0: [1, 2, 3] at indices 0, 1, 2
    // Column 1: [4, 5, 6] at indices 3, 4, 5
    // So data = [1, 2, 3, 4, 5, 6]
    
    let x = Tensor::from_f32(&[1.0, 2.0, 3.0], vec![3]).unwrap();
    let w = Tensor::from_f32(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![3, 2]).unwrap();
    let mut out = Tensor::zeros(vec![2], DType::F32);
    
    backend.vec_mat(&x, &w, &mut out).unwrap();
    
    let out_data = out.as_f32().unwrap();
    println!("x @ W = {:?}", out_data);
    println!("Expected: [14.0, 32.0]");
    
    assert!((out_data[0] - 14.0).abs() < 0.001, "First element should be 14");
    assert!((out_data[1] - 32.0).abs() < 0.001, "Second element should be 32");
    println!("Test PASSED!");
}
