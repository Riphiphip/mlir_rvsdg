func.func @test () -> f32 {
    %select_val = arith.constant 1: i32
    %predicate = rvsdg.match(%select_val : i32) [
            #rvsdg.matchRule<0, 1-> 1>,
            #rvsdg.matchRule<default -> 0>
    ] -> !rvsdg.ctrl<2>

    %test = rvsdg.constantCtrl 10 : !rvsdg.ctrl<2>

    %some_variable = arith.constant 14.0: f32
    %gammaResult0, %gammaResult1 = rvsdg.gammaNode (%predicate : !rvsdg.ctrl<2>) (%some_variable:f32):[
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