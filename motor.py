import numpy as np


class Motor:
    def __init__(self, battery_cells=3, duty_cycle=0.0):
        self.KV = 26.0
        self.MAX_TORQUE = 0.45
        self.INTERNAL_RESISTANCE = 15.0

        # Напряжения
        self.battery_voltage = 3.7 * battery_cells  # 11.1В
        self.Kt = 9.55 / self.KV

        self.max_current = 20.0
        self.friction_coeff = 0.0005

        # Состояние
        self.duty_cycle = duty_cycle

        # Для расчета RPM
        self.current_rpm = 0.0
        self._prev_angle_deg = None
        self._prev_time = None
        self._rpm_filter_alpha = 0.3

        # Фильтр момента
        self._torque_filtered = 0.0
        self._filter_alpha = 0.3

    def get_voltage(self):
        return self.duty_cycle * self.battery_voltage

    def calculate_current(self, rpm):
        U_eff = self.get_voltage()
        back_emf = rpm / self.KV
        delta_U = U_eff - back_emf
        current = delta_U / self.INTERNAL_RESISTANCE
        return np.clip(current, -self.max_current, self.max_current)

    def update_rpm(self, angle_rad, current_time):
        angle_deg = np.degrees(angle_rad)

        if self._prev_time is None or self._prev_angle_deg is None:
            self._prev_angle_deg = angle_deg
            self._prev_time = current_time
            return 0.0

        dt = current_time - self._prev_time
        if dt <= 0:
            return self.current_rpm

        delta_theta = angle_deg - self._prev_angle_deg
        if delta_theta > 180:
            delta_theta -= 360
        elif delta_theta < -180:
            delta_theta += 360

        deg_per_sec = delta_theta / dt
        rpm_raw = (deg_per_sec / 360.0) * 60.0

        self.current_rpm = (self._rpm_filter_alpha * rpm_raw +
                            (1 - self._rpm_filter_alpha) * self.current_rpm)

        self._prev_angle_deg = angle_deg
        self._prev_time = current_time

        return self.current_rpm

    def calculate_torque(self, rpm=None):
        if rpm is None:
            rpm = self.current_rpm

        current = self.calculate_current(rpm)

        # Момент от тока + трение
        torque = self.Kt * current - self.friction_coeff * rpm

        torque = np.clip(torque, -self.MAX_TORQUE, self.MAX_TORQUE)

        # Фильтр
        self._torque_filtered = (self._filter_alpha * torque +
                                 (1 - self._filter_alpha) * self._torque_filtered)

        return self._torque_filtered

    def set_duty_cycle(self, duty_cycle):
        self.duty_cycle = np.clip(duty_cycle, -1.0, 1.0)
