
rvsdg.omegaNode (%external_import: i32): {
    %test = arith.constant 1.0: f32
    
    rvsdg.omegaResult(%test:f32, %external_import: i32)
}
