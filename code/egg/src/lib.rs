use gag::Gag;
use egg::{rewrite as rw, *};
use std::collections::*;
use std::*;
use ordered_float::NotNan;
use pyo3::prelude::*;


// fn cost
static mut FN_COST: f64 = 1.0;
// flt fn cost
static mut FLT_FN_COST: f64 = 0.1;
// const weight
//static mut CONST_COST: f64 = 0.5;
static mut CONST_COST: f64 = 2.0;
// float var weight or float weight
static mut FLT_VAR_COST: f64 = 2.0;
// error weight
static mut ERR_WEIGHT: f64 = 10.;
// cat var weight
static mut CAT_COST: f64 = 0.5;

// Floats FNS
static mut FLT_FN_LIST: &'static mut [&str] = &mut ["Add", "Sub", "Mul", "Div", "Inv"];

// CAT CONSTS
static mut CAT_CONSTS: &'static mut [&str] = &mut ["AX","AY","AZ","INT#"];

// catch al
static mut DEF_COST: f64 = 1.0;

//static mut egrid_val_map: HashMap<Id, f64> = HashMap::new();

pub type Constant = NotNan<f64>;

define_language! {
    pub enum ShapeLNG {

        "Err" = Err([Id; 2]),
        Constant(Constant),
        Symbol(Symbol),
	Other(Symbol, Vec<Id>),
    }
}

pub struct ShapeLNGCostFn;
pub type EGraph = egg::EGraph<ShapeLNG, ShapeAnalysis>;
pub type Rewrite = egg::Rewrite<ShapeLNG, ShapeAnalysis>;


#[derive(Debug, Clone)]
pub struct SAData {
    egrid_val_map: HashMap<Id, f64>,
}

#[derive(Default)]
pub struct ShapeAnalysis;

fn get_def_evm() -> SAData {
   let mut egrid_val_map: HashMap<Id, f64> = HashMap::new();
   SAData { egrid_val_map }
}

impl Analysis<ShapeLNG> for ShapeAnalysis {
  type Data = SAData;

  fn make(egraph: &EGraph, enode: &ShapeLNG) -> Self::Data {
      return get_def_evm()
  }

  fn merge(&mut self, to: &mut Self::Data, from: Self::Data) -> DidMerge {

      if from.egrid_val_map.len() > 0 {
         println!("Unexpected behavior, please raise github issue if this persists");
         let mut egrid_val_map = from.egrid_val_map.clone();
         *to = SAData {egrid_val_map};
	 return DidMerge(true, false)
      } else {
        return DidMerge(false, false)
      }
  }

  fn modify(egraph: &mut EGraph, id: Id) {
    if id == Id::from(0) {
      assert_eq!(egraph[Id::from(0)].data.egrid_val_map.len(), 0);
      egraph[Id::from(0)].data.egrid_val_map.insert(Id::from(0), 10000000.0);
    }
  }

}

// Define costs of special symbols

fn symb_cost(enode: &ShapeLNG) -> f64 {
  unsafe {
  let name = enode.to_string();
  let mut split = name.split("_");
  let vec = split.collect::<Vec<&str>>();
  if name.contains("E_") {
    let v : f64 = vec[1].parse().unwrap();
    return v * ERR_WEIGHT;
  } else if name.contains("_C_") { 
    return CAT_COST;
  } else if name.contains("_F_") {
    return FLT_VAR_COST;
  }

  for cat_const in CAT_CONSTS.iter() {
    if name.contains(&*cat_const) {
       return CAT_COST
    }
  }
  
  println!("Unexpected token {}", name);
  return DEF_COST;
  
  }
}

// Define costs of functions

fn fn_cost(enode: &ShapeLNG) -> f64 {
  unsafe { 
  let name = enode.to_string();
  let why: &str = &*name;
  if FLT_FN_LIST.contains(&why) {
    return FLT_FN_COST;
  } else { 
    return FN_COST;
  }
  }
}

impl CostFunction<ShapeLNG> for ShapeLNGCostFn {
    type Cost = f64;
    fn cost<C>(&mut self, enode: &ShapeLNG, mut costs: C) -> Self::Cost
    where
        C: FnMut(Id) -> Self::Cost,
    {
        unsafe {
        let op_cost = match enode {
            ShapeLNG::Symbol(..) => symb_cost(enode),
            ShapeLNG::Constant(..) => CONST_COST,
   	    ShapeLNG::Err(..) => 0.0,	    
            ShapeLNG::Other(..) => fn_cost(enode),
        };
        enode.fold(op_cost, |sum, i| sum + costs(i))
	}
    }
}	

pub fn rules() -> Vec<Rewrite> { vec![]}


pub fn make_rules(ruleset : Vec<Vec<&str>>) -> Vec<Rewrite> {
    let mut def = rules();
    
    for rule in ruleset {
       let inner_apply = rule[2].parse::<Pattern<_>>().unwrap();

       if rule[3].contains("abs") {

          let mut split1 = rule[3].split(":");
          let vec1 = split1.collect::<Vec<&str>>();

	  let match_thresh : f64 = vec1[1].parse().unwrap();

          let mut expr_split = vec1[2].split(",");
          let expr_strings = expr_split.collect::<Vec<&str>>();
	  
          let outer_apply = ConditionalApplier {
             applier:inner_apply,
             condition:check_pattern(expr_strings, match_thresh)
          };

          def.push(Rewrite::new(
             rule[0],
  	     rule[1].parse::<Pattern<_>>().unwrap(),
	     outer_apply
          ).unwrap());

       } else if rule[3].contains("expr_record") {
 
          let mut split1 = rule[3].split(":");
          let vec1 = split1.collect::<Vec<&str>>();

	  let mut expr_string = vec1[1];
	  	  
          let outer_apply = ConditionalApplier {
             applier:inner_apply,
	     condition: do_expr_record(expr_string)
          };

          def.push(Rewrite::new(
             rule[0],
             rule[1].parse::<Pattern<_>>().unwrap(),
	     outer_apply
          ).unwrap());
     
       } else if rule[3].contains("var_record") {

         let mut split = rule[3].split(":");
         let vec = split.collect::<Vec<&str>>();
         let val : f64 = vec[1].parse().unwrap();

	 let outer_apply = ConditionalApplier {
             applier:inner_apply,
	     condition: do_var_record(val)
          };

         def.push(Rewrite::new(
           rule[0],
  	   rule[1].parse::<Pattern<_>>().unwrap(),
	   outer_apply
         ).unwrap());

       } else {
 
          def.push(Rewrite::new(
            rule[0],
  	    rule[1].parse::<Pattern<_>>().unwrap(),
 	    inner_apply
          ).unwrap());
       }       
    }

    return def;
}

fn egg_sc_extract(
  expr : &str, once_ruleset : Vec<Vec<&str>>, sat_ruleset : Vec<Vec<&str>>, iter_limit : usize, node_limit : usize, time_limit : u64
) -> String {
  let start = expr.parse().unwrap();

  let once_rls:  &[Rewrite] = &make_rules(once_ruleset);
  let sat_rls:  &[Rewrite] = &make_rules(sat_ruleset);

  let mut runner = Runner::default()
    .with_iter_limit(1)
    .with_node_limit(node_limit)
    .with_time_limit(std::time::Duration::from_secs(time_limit))		
    .with_expr(&start)
    .with_scheduler(SimpleScheduler)
    .run(once_rls);

  runner.stop_reason = None;

  let sat_runner = runner
    .with_iter_limit(iter_limit)
    .with_node_limit(node_limit)
    .with_time_limit(std::time::Duration::from_secs(time_limit))		
    .with_expr(&start)
    .with_scheduler(SimpleScheduler)
    .run(sat_rls);

  let root = sat_runner.egraph.find(sat_runner.roots[0]);

  let GreedExtractor = Extractor::new(&sat_runner.egraph, ShapeLNGCostFn);  
  let (cost, greed_expr) = GreedExtractor.find_best(root);

  let tog = format!("{}+{}+{:?}", greed_expr, cost, sat_runner.stop_reason);  

  tog
}

#[pyfunction]
fn sc_extract(expr : &str, once_ruleset : Vec<Vec<&str>>, sat_ruleset : Vec<Vec<&str>>, iter_limit : usize, node_limit : usize, time_limit : u64) -> PyResult<String> {
  Ok(egg_sc_extract(expr, once_ruleset, sat_ruleset, iter_limit, node_limit, time_limit))
}

#[pymodule]
fn sc_egraph(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(sc_extract, m)?)?;
    Ok(())
}


fn do_var_record(val : f64) -> impl Fn(&mut EGraph, Id, &Subst) -> bool  {
    move |egraph, id, subst| {
        egraph[Id::from(0)].data.egrid_val_map.insert(id, val);        
        false
    }
}

fn do_expr_record(expr_str : &str)
  -> impl Fn(&mut EGraph, Id, &Subst) -> bool  {
  
    let exfn = parse_expr_fn(expr_str.to_string());

    move |egraph, id, subst| {

        let val = (exfn)(egraph, subst);

        egraph[Id::from(0)].data.egrid_val_map.insert(id, val);

        false
    }
}

pub struct FEXStruct<F>
  where F: Fn(&mut EGraph, &Subst) -> bool
{
  fn_list: Vec<F>
}

impl <F> FEXStruct<F>
  where F: Fn(&mut EGraph, &Subst) -> bool

{
  pub fn new() -> FEXStruct<F> {
     FEXStruct {
       fn_list : Vec::new(),
     }
  }

  pub fn add_fn(&mut self, func: F) {
    self.fn_list.push(func);
  }
}

fn h_parse_expr_fn(math_string: &str, egraph: &mut EGraph, subst: &Subst) -> f64 {

  let operations = math_string.split("+");

  let mut regs = vec![0.0, 0.0, 0.0];
  
  let mut _data : f64 = 0.0;

  for op_ord in operations {
     let op_res = op_ord.split("|");
     let op_info = op_res.collect::<Vec<&str>>();
     
     let cmd = op_info[0];
     let ind = op_info[1];
     let Vreg = op_info[2];

     if ind.contains("reg") {
        let reg_split = ind.split('_');
	let reg_col = reg_split.collect::<Vec<&str>>();
     	let ri : usize = reg_col[1].parse().unwrap();
       _data = regs[ri];
     } else if ind.contains("?") {
       let vn : Var = ind.parse().unwrap();
       _data = egraph[Id::from(0)].data.egrid_val_map[&subst[vn]];
     } else {
       _data = ind.parse().unwrap();
     }

     let reg_ind : usize = Vreg.parse().unwrap();

     if cmd.contains("new") {
       regs[reg_ind] = _data;
     } else if cmd.contains("Add") {
       regs[reg_ind] += _data;
     } else if cmd.contains("Sub") {
       regs[reg_ind] -= _data;
     } else if cmd.contains("Mul") {
       regs[reg_ind] *= _data;
     } else if cmd.contains("Div") {
       regs[reg_ind] /= _data;
     } else if cmd.contains("Inv") {
       regs[reg_ind] *= -1.0;
     }    
  }
  regs[0]
}



fn parse_expr_fn(math_string: String) -> impl Fn(&mut EGraph, &Subst) -> f64 {
   
   move |egraph : &mut EGraph, subst: &Subst| {
      h_parse_expr_fn(math_string.as_str(), egraph, subst)	            
   }
}



  

fn make_expr_fn(expr_string : &str, match_thresh : f64) -> impl Fn(&mut EGraph, &Subst) -> bool {

    let mut split = expr_string.split("=");
    let vec = split.collect::<Vec<&str>>();
    let tar_name : Var = vec[0].parse().unwrap();

    let expr_fn = parse_expr_fn(vec[1].to_string());
    
    move |egraph : &mut EGraph, subst: &Subst| {
        let tar_val = egraph[Id::from(0)].data.egrid_val_map[&subst[tar_name]];
	let expr_val = (expr_fn)(egraph, subst);
	let pass = (tar_val - expr_val).abs() <= match_thresh;
	pass
    }

}

fn check_pattern(expr_strings : Vec<&str>, match_thresh : f64) -> impl Fn(&mut EGraph, Id, &Subst) -> bool {

    let mut fexs = FEXStruct::new();
   
    for expr_string in expr_strings {
       if expr_string.len() >= 1 {
          let exfn = make_expr_fn(expr_string, match_thresh);
          fexs.add_fn(exfn);
       }	
    }   
   
    move |egraph, id, subst| {
        for efn in &fexs.fn_list {
	  let pass = (efn)(egraph, subst);
	  if !pass {
	     return false
	  }
 	}	
	true
    }
}
