func.func @test () -> f32 {
    %predicate = arith.constant 0: index
    %some_variable = arith.constant 14.0: f32
    %gammaResult0, %gammaResult1 = rvsdg.gammaNode (%predicate) (%some_variable:f32):[
        (%some_variable: f32):{
            %c = arith.constant 1.0: f32
            %d = arith.constant 20: i32
            rvsdg.gammaResult (%c:f32, %some_variable:f32)
        },
        (%some_variable: f32):{
            %c = arith.constant 2.0: f32
            %d = arith.constant 40: i32
            rvsdg.gammaResult (%c:f32, %c:f32)
        }
    ] -> f32, f32
    func.return %gammaResult0: f32
}