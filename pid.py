import numpy as np
import pybullet as p
import pybullet_data
import os
import time

from motor import Motor

tau_max = 0.45
g = 9.81

def wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


def alpha(theta):
    return wrap(theta - np.pi)


class PID:
    def __init__(self):
        self.int_t = 0.0
        self.int_a = 0.0
        self.prev_error_t = 0.0
        self.prev_error_a = 0.0

        # Коэффициенты для маятника
        self.Kp_t = 80.0
        self.Ki_t = 5.0
        self.Kd_t = 12.0

        # Коэффициенты для горизонтального вала
        self.Kp_a = 8.0
        self.Ki_a = 2.0
        self.Kd_a = 3.0

        self.int_limit_t = 10.0
        self.int_limit_a = 5.0

    def control(self, th1, th2, dth1, dth2, dt, theta_ref=0.0):
        # Ошибка маятника
        angle_error = wrap(th2)

        u_t = (self.Kp_t * angle_error +
               self.Ki_t * self.int_t -
               self.Kd_t * dth2)

        u_t_limited = np.clip(u_t, -tau_max, tau_max)

        if abs(u_t) < tau_max * 0.95:
            self.int_t += angle_error * dt
            self.int_t = np.clip(self.int_t, -self.int_limit_t, self.int_limit_t)

        arm_error = wrap(th1 - theta_ref)

        u_a = (self.Kp_a * arm_error +
               self.Ki_a * self.int_a -
               self.Kd_a * dth1)

        u_a_limited = np.clip(u_a, -tau_max, tau_max)

        if abs(u_a) < tau_max * 0.95:
            self.int_a += arm_error * dt
            self.int_a = np.clip(self.int_a, -self.int_limit_a, self.int_limit_a)

        u = u_t_limited + u_a_limited
        u = np.clip(u, -tau_max, tau_max)

        return float(u)


def main():
    p.connect(p.GUI)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, -g)

    dt = 1.0 / 1000.0
    p.setTimeStep(dt)
    p.loadURDF("plane.urdf")

    urdf = os.path.join(os.path.dirname(__file__), "furuta_pendulum.urdf")
    robot = p.loadURDF(urdf, [0, 0, 0.3], useFixedBase=True)

    ARM, PEND = 0, 1

    p.setJointMotorControl2(robot, ARM, p.VELOCITY_CONTROL, force=0)
    p.setJointMotorControl2(robot, PEND, p.VELOCITY_CONTROL, force=0)

    p.resetJointState(robot, PEND, np.pi-0.1)
    p.resetJointState(robot, ARM, 0.0)

    p.resetDebugVisualizerCamera(
        cameraDistance=0.35,
        cameraYaw=35,
        cameraPitch=-25,
        cameraTargetPosition=[0, 0, 0.35]
    )

    pid = PID()
    motor = Motor()

    theta_ref = 0.0

    step = 0

    try:
        while p.isConnected():
            th1, dth1, _, _ = p.getJointState(robot, ARM)
            th2, dth2, _, _ = p.getJointState(robot, PEND)

            # Отклонение от вертикали
            a_rad = alpha(th2)
            a_deg = np.degrees(a_rad)

            # Обновление RPM мотора
            current_time = step * dt
            motor.update_rpm(wrap(th1), current_time)

            if abs(a_deg) < 20.0:
                u = pid.control(th1, th2, dth1, dth2, dt, theta_ref)

                # Преобразование в duty cycle
                duty = np.clip(u / tau_max, -1.0, 1.0)
                motor.set_duty_cycle(duty)
                torque = motor.calculate_torque()

            else:
                u = 0.0
                duty = 0.0
                torque = 0.0
                motor.set_duty_cycle(0.0)

            p.setJointMotorControl2(
                robot,
                ARM,
                p.TORQUE_CONTROL,
                force=float(torque)
            )

            p.stepSimulation()

            step += 1
            time.sleep(dt)

    except KeyboardInterrupt:
        print("\n\nОстановка")

    p.setJointMotorControl2(robot, ARM, p.TORQUE_CONTROL, force=0)
    p.disconnect()


if __name__ == "__main__":
    main()