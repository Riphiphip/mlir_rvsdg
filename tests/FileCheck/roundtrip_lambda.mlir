
%4 = arith.constant 4: i32
%5 = arith.constant 4.0: f32

%6 = rvsdg.lambdaNode <(i32)->(f32)> (%4:i32, %5:f32):
    (%arg1: i32, %test: i32, %aasd:f32): {
    %0 = arith.constant 1.0: f32
    rvsdg.lambdaResult(%0:f32)
}

%100 = rvsdg.applyNode %6:<(i32)->(f32)>(%4:i32) -> f32