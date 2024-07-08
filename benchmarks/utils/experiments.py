from abc import ABC, abstractmethod

# Add the root folder to the path
import sys
import os
relative_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(relative_path + "/../../")

import numpy as np
import power_alm.functions as fun
import power_alm.inner_solver as optim
import power_alm.solver as solver

import matplotlib.pyplot as plt

from pathlib import Path

RESULTS = "results/"

class Experiment(ABC):
    def __init__(self, name, n, config, optimizer_configs, num_runs):
        self.name = name
        self.config = config
        self.optimizer_configs = optimizer_configs
        self.num_runs = num_runs
        self.n = n

        #assigned by subclass
        self.composite_problem = None
        np.random.seed(config["seed"])

        self.initialize_composite_problem()

    @abstractmethod
    def compute_optimum(self, x_init):
        pass

    @abstractmethod
    def get_filename(self):
        pass

    @abstractmethod
    def eval_objective(self, x):
        pass

    @abstractmethod
    def eval_constraint_violation(self, x):
        pass

    @abstractmethod
    def eval_power_stepsize(self, x):
        pass

    @abstractmethod
    def initialize_composite_problem(self):
        pass

    def run(self, overwrite_file=False):
        filename = RESULTS + self.get_filename()
        suffix = ".npz"

        if overwrite_file or not Path(filename + suffix).is_file():
            self.compute_optimum(np.zeros(self.n))

            self.objective_values = dict()
            self.constraint_violations = dict()
            self.nit = dict()
            self.power_stepsizes = dict()

            for optimizer_config in self.optimizer_configs:
                self.objective_values[optimizer_config["name"]] = [[] for i in range(self.num_runs)]
                self.constraint_violations[optimizer_config["name"]] = [[] for i in range(self.num_runs)]
                self.nit[optimizer_config["name"]] = [[] for i in range(self.num_runs)]
                self.power_stepsizes[optimizer_config["name"]] = [[] for i in range(self.num_runs)]

            fun.counting_enabled = True

            for run in range(self.num_runs):
                # Generate new random problem realization
                if run > 0:
                    self.initialize_composite_problem()
                    self.compute_optimum(np.zeros(self.n))

                x_init = eval(self.config["init_proc"] + "(" + str(self.n) + ")")
                problem = self.composite_problem

                for optimizer_config in self.optimizer_configs:
                    print(optimizer_config["name"])

                    def callback(k, nit, x):

                        self.objective_values[optimizer_config["name"]][run].append(np.abs(self.eval_objective(x) - self.opt_obj))
                        self.constraint_violations[optimizer_config["name"]][run].append(self.eval_constraint_violation(x) - self.opt_const)
                        self.nit[optimizer_config["name"]][run].append(nit)
                        self.power_stepsizes[optimizer_config["name"]][run].append(self.eval_power_stepsize(x))

                        if (k-1)%50 == 0 or True:
                            print(k, nit, self.eval_objective(x) - self.opt_obj, self.eval_constraint_violation(x) - self.opt_const, np.linalg.norm(self.composite_problem.diffable.eval_gradient(x)))
                        
                        if np.abs(self.eval_objective(x) - self.opt_obj) < (optimizer_config["parameters"]).tol and np.abs(self.eval_constraint_violation(x) - self.opt_const) < (optimizer_config["parameters"]).tol:
                            return True
                        else:
                            return False

                    optimizer = solver.Solver(
                        problem, 
                        optimizer_config["parameters"], 
                        inner_solver=optimizer_config["class"],
                        callback=callback
                    )

                    optimizer.run()

                    print(optimizer.params.maxit, self.eval_objective(optimizer.x) - self.opt_obj, self.eval_constraint_violation(optimizer.x) - self.opt_const)

            np.savez(filename, 
                     opt_obj=self.opt_obj,
                     objective_values=self.objective_values,
                     constraint_violations=self.constraint_violations,
                     nit=self.nit,
                     power_stepsizes=self.power_stepsizes,
            )

        else:
            cache = np.load(filename + '.npz', allow_pickle=True)
            self.opt_obj = cache["opt_obj"]
            self.nit = cache.get("nit").item()
            self.constraint_violations = cache["constraint_violations"].item()
            self.objective_values = cache["objective_values"].item()
            self.power_stepsizes = cache["power_stepsizes"].item()


    def plot(self, markevery, SINGLE_PLOT = False, PLOT_TOTAL_NIT = True, run = None, xlabels = None, ylabels = None):
        if self.num_runs == 1:
            run = 0
            
        for optimizer_config in self.optimizer_configs:
            if run is not None:
                if not SINGLE_PLOT:
                    plt.subplot(2, 1, 1)
                if PLOT_TOTAL_NIT:
                    plt.semilogy(
                        self.nit[optimizer_config["name"]][run],
                        np.array(self.objective_values[optimizer_config["name"]][run]),
                        label=optimizer_config["label"],
                        marker=optimizer_config["marker"],
                        markevery=markevery,
                        color=optimizer_config["color"],
                        linestyle=optimizer_config["linestyle"]
                    )
                else:
                    plt.semilogy(
                        np.array(self.objective_values[optimizer_config["name"]][run]),
                        label=optimizer_config["label"],
                        marker=optimizer_config["marker"],
                        markevery=markevery,
                        color=optimizer_config["color"],
                        linestyle=optimizer_config["linestyle"]
                    )

                if xlabels is not None:
                    plt.xlabel(xlabels[0])
                else:
                    plt.xlabel("iteration $k$")
                if ylabels is not None:
                    plt.ylabel(ylabels[0])
                else:
                    plt.ylabel("$\\varphi(x) - \\varphi(x^\\star)$")

                if not SINGLE_PLOT:
                    plt.subplot(2, 1, 2)
                    if PLOT_TOTAL_NIT:
                        plt.semilogy(
                            self.nit[optimizer_config["name"]][run],
                            np.array(self.constraint_violations[optimizer_config["name"]][run]),
                            label=optimizer_config["label"],
                            marker=optimizer_config["marker"],
                            markevery=markevery,
                            color=optimizer_config["color"],
                            linestyle=optimizer_config["linestyle"]
                        )
                    else:
                        plt.semilogy(
                            np.array(self.constraint_violations[optimizer_config["name"]][run]),
                            label=optimizer_config["label"],
                            marker=optimizer_config["marker"],
                            markevery=markevery,
                            color=optimizer_config["color"],
                            linestyle=optimizer_config["linestyle"]
                        )
                    plt.xlabel("iteration $k$")
                    if ylabels is not None:
                        plt.ylabel(ylabels[1])
                    else:
                        plt.ylabel("Constraint violation")
            else:
                raise RuntimeError("This experiment contains multiple runs. Please specify which run to plot.")


        plt.tight_layout()
        plt.legend()

    def compute_grid(self, MEDIANS_ONLY = True, P95S_ONLY = False, SAVE_CSV = False, filename = None):
        assert self.num_runs > 1

        medians = dict()
        p95s = dict()

        for optimizer_config in self.optimizer_configs:
            nit = np.array(list(map(lambda x : x[-1], self.nit[optimizer_config["name"]])))
            medians[optimizer_config["name"]] = np.median(nit)
            p95s[optimizer_config["name"]] = np.percentile(nit, 95)

        print()
        logs = np.zeros(len(self.optimizer_configs))
        for i, optimizer_config in enumerate(self.optimizer_configs):
            if MEDIANS_ONLY:
                print(f'{medians[optimizer_config["name"]]}\t'.expandtabs(16), end='')
                logs[i] = medians[optimizer_config["name"]]
            elif P95S_ONLY:
                print(f'{p95s[optimizer_config["name"]]}\t'.expandtabs(16), end='')
                logs[i] = p95s[optimizer_config["name"]]
            else:
                print(f'{optimizer_config["name"]}:\t{medians[optimizer_config["name"]]},\t{p95s[optimizer_config["name"]]}'.expandtabs(16))

        print()

        if SAVE_CSV:
            assert filename is not None
            with open(RESULTS + filename + ".csv", "a") as f:
                np.savetxt(f, np.array([logs]))
                # f.write("\n")

    def plot_powerstepsizes(self, markevery, PLOT_TOTAL_NIT = True, run = None, xlabels = None, ylabels = None):
        if self.num_runs == 1:
            run = 0
            
        for optimizer_config in self.optimizer_configs:
            if run is not None:
                if PLOT_TOTAL_NIT:
                    plt.semilogy(
                        self.nit[optimizer_config["name"]][run],
                        np.array(self.power_stepsizes[optimizer_config["name"]][run]),
                        label=optimizer_config["label"],
                        marker=optimizer_config["marker"],
                        markevery=markevery,
                        color=optimizer_config["color"],
                        linestyle=optimizer_config["linestyle"]
                    )
                else:
                    plt.semilogy(
                        np.array(self.power_stepsizes[optimizer_config["name"]][run]),
                        label=optimizer_config["label"],
                        marker=optimizer_config["marker"],
                        markevery=markevery,
                        color=optimizer_config["color"],
                        linestyle=optimizer_config["linestyle"]
                    )

                if xlabels is not None:
                    plt.xlabel(xlabels[0])
                else:
                    plt.xlabel("iteration $k$")
                if ylabels is not None:
                    plt.ylabel(ylabels[0])
                else:
                    plt.ylabel("Penalty")
            else:
                raise RuntimeError("This experiment contains multiple runs. Please specify which run to plot.")


        plt.tight_layout()
        plt.legend(loc = "lower right")