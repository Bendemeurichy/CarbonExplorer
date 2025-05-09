# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the CC-BY-NC license found in the
# LICENSE file in the root directory of this source tree.

import pandas as pd


def cas_binary(
    df_all: pd.DataFrame,
    flexible_workload_ratio: float,
    max_capacity: float,
):
    """
    Carbon Aware Scheduling Algorithm using binary search.
    Optimizes for 24/7 carbon-aware scheduling.

    Args:
        df_all (pd.DataFrame): Dataframe containing renewable and dc power data.
        flexible_workload_ratio (float): Ratio of flexible workload to be moved.
        max_capacity (float): Maximum capacity for the scheduling.

    Returns:
        pd.DataFrame: Carbon balanced version of the input dataframe.
    """
    # Implement binary search logic here
    # This is a placeholder implementation
    balanced_df = []

    for i in range(0, df_all.shape[0], 24):
        daily_df = df_all[i : i + 24].copy()
        if daily_df.shape[0] < 24:
            break

        # Calculate total renewable deficit for the day
        renewable_deficit = []
        for j in range(daily_df.shape[0]):
            deficit = max(
                0,
                daily_df["avg_dc_power_mw"].iloc[j] - daily_df["tot_renewable"].iloc[j],
            )
            renewable_deficit.append((j, deficit))

        # Sort hours by renewable deficit (descending)
        renewable_deficit.sort(key=lambda x: x[1], reverse=True)

        # Calculate total movable workload
        total_movable = sum(daily_df["avg_dc_power_mw"]) * flexible_workload_ratio / 100
        workload_moved = 0

        # Binary search for optimal threshold
        left = 0
        right = max([deficit for _, deficit in renewable_deficit])

        while left <= right:
            mid = (left + right) / 2

            # Try applying this threshold
            test_df = daily_df.copy()
            deficit_hours = [idx for idx, deficit in renewable_deficit if deficit > mid]
            surplus_hours = [
                j for j in range(daily_df.shape[0]) if j not in deficit_hours
            ]

            # Sort surplus hours by maximum available renewable capacity
            surplus_hours.sort(
                key=lambda j: test_df["tot_renewable"].iloc[j]
                - test_df["avg_dc_power_mw"].iloc[j],
                reverse=True,
            )

            moved = 0
            for from_idx in deficit_hours:
                if moved >= total_movable:
                    break

                # Calculate how much can be moved from this hour
                movable = min(
                    test_df["avg_dc_power_mw"].iloc[from_idx]
                    * flexible_workload_ratio
                    / 100,
                    total_movable - moved,
                )

                for to_idx in surplus_hours:
                    available_space = min(
                        test_df["tot_renewable"].iloc[to_idx]
                        - test_df["avg_dc_power_mw"].iloc[to_idx],
                        max_capacity - test_df["avg_dc_power_mw"].iloc[to_idx],
                    )

                    if available_space <= 0:
                        continue

                    amount_to_move = min(movable, available_space)
                    if amount_to_move <= 0:
                        continue

                    test_df["avg_dc_power_mw"].iloc[from_idx] -= amount_to_move
                    test_df["avg_dc_power_mw"].iloc[to_idx] += amount_to_move
                    movable -= amount_to_move
                    moved += amount_to_move

                    if movable <= 0:
                        break

            if moved >= total_movable * 0.95:  # Allow 5% tolerance
                # This threshold works, try a higher one (more selective)
                left = mid + 0.1
            else:
                # This threshold is too high, try a lower one
                right = mid - 0.1

        balanced_df.append(test_df)

    final_balanced_df = pd.concat(balanced_df).sort_values(by=["index"])
    return final_balanced_df


def binary_cas_grid_mix(df_all, flexible_workload_ratio, max_capacity):
    """Binary search approach for carbon intensity optimization"""
    balanced_df = []

    for i in range(0, df_all.shape[0], 24):
        daily_df = df_all[i : i + 24].copy()
        if daily_df.shape[0] < 24:
            break

        # Sort hours by carbon intensity
        hourly_carbon = [
            (j, daily_df["carbon_intensity"].iloc[j]) for j in range(daily_df.shape[0])
        ]
        hourly_carbon.sort(key=lambda x: x[1], reverse=True)  # Highest carbon first

        # Calculate total movable workload
        total_movable = sum(daily_df["avg_dc_power_mw"]) * flexible_workload_ratio / 100

        # Binary search for optimal carbon reduction
        left = 0
        right = max([intensity for _, intensity in hourly_carbon]) - min(
            [intensity for _, intensity in hourly_carbon]
        )

        best_df = daily_df.copy()
        best_carbon_reduction = 0

        while right - left > 0.1:
            mid = (left + right) / 2

            test_df = daily_df.copy()
            carbon_threshold = min([intensity for _, intensity in hourly_carbon]) + mid

            from_hours = [
                idx for idx, intensity in hourly_carbon if intensity > carbon_threshold
            ]
            to_hours = [
                idx for idx, intensity in hourly_carbon if intensity <= carbon_threshold
            ]

            # Sort destination hours by carbon intensity (ascending)
            to_hours.sort(key=lambda j: test_df["carbon_intensity"].iloc[j])

            moved = 0
            carbon_reduced = 0

            for from_idx in from_hours:
                if moved >= total_movable:
                    break

                movable = min(
                    test_df["avg_dc_power_mw"].iloc[from_idx]
                    * flexible_workload_ratio
                    / 100,
                    total_movable - moved,
                )

                for to_idx in to_hours:
                    available_space = (
                        max_capacity - test_df["avg_dc_power_mw"].iloc[to_idx]
                    )

                    if available_space <= 0:
                        continue

                    amount_to_move = min(movable, available_space)
                    if amount_to_move <= 0:
                        continue

                    carbon_delta = amount_to_move * (
                        test_df["carbon_intensity"].iloc[from_idx]
                        - test_df["carbon_intensity"].iloc[to_idx]
                    )

                    # Move workload
                    test_df["avg_dc_power_mw"].iloc[from_idx] -= amount_to_move
                    test_df["avg_dc_power_mw"].iloc[to_idx] += amount_to_move
                    movable -= amount_to_move
                    moved += amount_to_move
                    carbon_reduced += carbon_delta

                    if movable <= 0:
                        break

            if carbon_reduced > best_carbon_reduction:
                best_df = test_df.copy()
                best_carbon_reduction = carbon_reduced

            # Adjust search space
            if moved >= total_movable * 0.95:  # 5% tolerance
                # We can move more workload, try a more aggressive threshold
                left = mid
            else:
                # Too aggressive, reduce threshold
                right = mid

        balanced_df.append(best_df)

    final_balanced_df = pd.concat(balanced_df).sort_values(by=["index"])
    return final_balanced_df
