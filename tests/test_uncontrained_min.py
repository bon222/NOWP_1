import unittest
import examples
from examples import *
import sys
sys.path.insert(1,'/Users/aaron/Documents/GitHub/NOWP_1/src')
import unconstrained_min 
from unconstrained_min import *
import utils
from utils import *

class TestMinimization(unittest.TestCase):
  #     quad_1_gd = minimize(quadratic_1,"grad_desc")
#     utils.plot_contour(quadratic_1, limits = [-5, 5, -5, 5], heading="quadratic 1 with gradient descennt",vectors=quad_1_gd[2])
 #       plot_function_values(quad_1_gd[3],"quadratic 1 with gradient descent")
#
 #   def test_quadratic_1_newton(self):
  #     plot_contour(quadratic_1, limits = [-5, 5, -5, 5], heading="quadratic 1 with newton",vectors=quad_1_new[2])
  #      plot_function_values(quad_1_new[3],"quadratic 1 with newton")
  #  
  #  def test_quadratic_1_sr1(self):
  #      quad_1_sr1 = minimize(quadratic_1,"SR1")
   #     plot_contour(quadratic_1, limits = [-5, 5, -5, 5], heading="quadratic 1 with SR1",vectors=quad_1_sr1[2])
  #      plot_function_values(quad_1_sr1[3],"quadratic 1 with SR1")

  #  def test_quadratic_1_bfgs(self):
  #      quad_1_bfgs = minimize(quadratic_1,"BFGS")
  #      plot_contour(quadratic_1, limits = [-5, 5, -5, 5], heading="quadratic 1 with BFGS",vectors=quad_1_bfgs[2])
   #     plot_function_values(quad_1_bfgs[3],"quadratic 1 with BFGS")

  #  def test_quadratic_2_grad_desc(self):
  #      quad_2_gd = unconstrained_min.minimize(quadratic_2,"grad_desc")
   #     utils.plot_contour(quadratic_2, limits = [-5, 5, -5, 5], heading="quadratic 2 with gradient descennt",vectors=quad_2_gd[2])
   #     plot_function_values(quad_2_gd[3],"quadratic 2 with gradient descent")
        
   # def test_quadratic_2_newton(self):
   #     quad_2_new = unconstrained_min.minimize(quadratic_2,"newton")
   #     utils.plot_contour(quadratic_2, limits = [-5, 5, -5, 5], heading="quadratic 2 with newton",vectors=quad_2_new[2])
   #     plot_function_values(quad_2_new[3],"quadratic 2 with newton")

    #def test_quadratic_2_sr1(self):
    #    quad_2_sr1 = minimize(quadratic_2,"SR1")
    #    plot_contour(quadratic_2, limits = [-5, 5, -5, 5], heading="quadratic 2 with SR1",vectors=quad_2_sr1[2])
      #  plot_function_values(quad_2_sr1[3],"quadratic 2 with SR1")

   # def test_quadratic_2_bfgs(self):
     #   quad_2_bfgs = minimize(quadratic_2,"BFGS")
     #   plot_contour(quadratic_2, limits = [-5, 5, -5, 5], heading="quadratic 2 with BFGS",vectors=quad_2_bfgs[2])
      #  plot_function_values(quad_2_bfgs[3],"quadratic 2 with BFGS")

   # def test_quadratic_3_grad_desc(self):
   #     quad_3_gd = unconstrained_min.minimize(quadratic_3,"grad_desc")
    #    utils.plot_contour(quadratic_3, limits = [-5, 5, -5, 5], heading="quadratic 3 with gradient descennt",vectors=quad_3_gd[2])
    #    plot_function_values(quad_3_gd[3],"quadratic 3 with gradient descent")

   # def test_quadratic_3_newton(self):
   #     quad_3_new = unconstrained_min.minimize(quadratic_3,"newton")
    #    utils.plot_contour(quadratic_3, limits = [-5, 5, -5, 5], heading="quadratic 3 with newton",vectors=quad_3_new[2])
     #   plot_function_values(quad_3_new[3],"quadratic 3 with gradient descent")

    #def test_quadratic_3_sr1(self):
     #   quad_3_sr1 = minimize(quadratic_3,"SR1")
       # plot_contour(quadratic_3, limits = [-5, 5, -5, 5], heading="quadratic 3 with SR1",vectors=quad_3_sr1[2])
      #  plot_function_values(quad_3_sr1[3],"quadratic 3 with SR1")

   # def test_quadratic_3_bfgs(self):
    #    quad_3_bfgs = minimize(quadratic_3,"BFGS")
     #   plot_contour(quadratic_3, limits = [-5, 5, -5, 5], heading="quadratic 3 with BFGS",vectors=quad_3_bfgs[2])
    #    plot_function_values(quad_3_bfgs[3],"quadratic 3 with BFGS")

   # def test_rosenbrock_gd(self):
   #     ros_gd = minimize(rosenbrock,"grad_desc",x0=np.array([-2,1]), max_iter=10000)
   #     utils.plot_contour(rosenbrock, limits = [-5, 5, -5, 5], heading="rosenbrock with gradient descent",vectors=ros_gd[2])
    #    plot_function_values(ros_gd[3],"rosenbrock with gradient descent")

    def test_rosenbrock_newton(self):
        ros_new = minimize(rosenbrock,"newton",x0=np.array([-2,1]))
        utils.plot_contour(rosenbrock, limits = [-5, 5, -5, 5], heading="rosenbrock with newton",vectors=ros_new[2])
        plot_function_values(ros_new[3],"rosenbrock with newton")

    def test_rosenbrock_sr1(self):
        ros_sr1 = minimize(rosenbrock,"SR1",x0=np.array([-2,1]))
        utils.plot_contour(rosenbrock, limits = [-15, 15, -15, 15], heading="rosenbrock with SR1",vectors=ros_sr1[2])
        plot_function_values(ros_sr1[3],"rosenbrock with SR1")

    def test_rosenbrock_bfgs(self):
        ros_bfgs = minimize(rosenbrock,"BFGS",x0=np.array([-2,1]))
        utils.plot_contour(rosenbrock, limits = [-15, 15, -15, 15], heading="rosenbrock with BFGS",vectors=ros_bfgs[2])
        plot_function_values(ros_bfgs[3],"rosenbrock with BFGS")

    def test_linear_gd(self):
        lin_gd = minimize(linear,"grad_desc")
        utils.plot_contour(linear, limits = [-5, 5, -5, 5], heading="linear with gradient descent",vectors=lin_gd[2])
        plot_function_values(lin_gd[3],"Linear with gradient descent")

    def test_linear_newton(self):
        lin_new = minimize(linear,"newton")
        utils.plot_contour(linear, limits = [-5, 5, -5, 5], heading="Linear with newton",vectors=lin_new[2])
        plot_function_values(lin_new[3],"Linear with newton")

    def test_linear_sr1(self):
        lin_sr1 = minimize(linear,"SR1")
        utils.plot_contour(linear, limits = [-5, 5, -5, 5], heading="linear with SR1",vectors=lin_sr1[2])
        plot_function_values(lin_sr1[3],"linear with SR1")

    def test_linear_bfgs(self):
        lin_bfgs = minimize(linear,"BFGS")
        utils.plot_contour(linear, limits = [-5, 5, -5, 5], heading="linear with BFGS",vectors=lin_bfgs[2])
        plot_function_values(lin_bfgs[3],"linear with BFGS")
    
    def test_triangle(self):
        plot_contour(triangles, limits = [-750, 750, -750, 750], heading="triangles contours",vectors=[])
    #def test_triangles_gd(self):
    #    tri_gd = minimize(triangles,"grad_desc")
    #    utils.plot_contour(triangles, limits = [-5, 5, -5, 5], heading="triangles with gradient descent",vectors=tri_gd[2])
    #    plot_function_values(tri_gd[3],"Triangles with gradient descent")

    def test_triangles_newton(self):
        tri_new = minimize(triangles,"newton")
        utils.plot_contour(triangles, limits = [-5, 5, -5, 5], heading="Triangles with newton",vectors=tri_new[2])
        plot_function_values(tri_new[3],"Triangles with newton")

    def test_triangles_sr1(self):
        tri_sr1 = minimize(triangles,"SR1")
        utils.plot_contour(triangles, limits = [-5, 5, -5, 5], heading="triangles with SR1",vectors=tri_sr1[2])
        plot_function_values(tri_sr1[3],"triangles with SR1")

    def test_triangles_bfgs(self):
        tri_bfgs = minimize(triangles,"BFGS")
        utils.plot_contour(triangles, limits = [-5, 5, -5, 5], heading="triangles with BFGS",vectors=tri_bfgs[2])
        plot_function_values(tri_bfgs[3],"triangles with BFGS")
if __name__ == "__main__":
    unittest.main()