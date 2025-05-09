# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the CC-BY-NC license found in the
# LICENSE file in the root directory of this source tree.

import pandas as pd


def cas_hybrid(df_all, flexible_workload_ratio, max_capacity):
    """Hybrid search approach for 24/7 renewable optimization using exponential followed by binary search"""
    balanced_df = []

    for i in range(0, df_all.shape[0], 24):
        daily_df = df_all[i : i + 24].copy()
        if daily_df.shape[0] < 24:
            break

        # Phase 1: Exponential search to find rough bounds
        renewable_deficit = []
        for j in range(daily_df.shape[0]):
            deficit = max(
                0,
                daily_df["avg_dc_power_mw"].iloc[j] - daily_df["tot_renewable"].iloc[j],
            )
            renewable_deficit.append((j, deficit))

        renewable_deficit.sort(key=lambda x: x[1], reverse=True)
        max_deficit = max([deficit for _, deficit in renewable_deficit])
        target_movable = (
            sum(daily_df["avg_dc_power_mw"]) * flexible_workload_ratio / 100
        )

        # Exponential search to find upper bound
        threshold = 0.1
        prev_threshold = 0
        while threshold < max_deficit:
            movable = 0
            for _, deficit in renewable_deficit:
                if deficit > threshold:
                    movable += deficit - threshold

            if movable < target_movable:
                # Found upper bound, break
                break

            prev_threshold = threshold
            threshold *= 2

        # Phase 2: Binary search to refine threshold
        left = prev_threshold
        right = threshold
        final_threshold = 0

        for _ in range(
            7
        ):
            mid = (left + right) / 2

            # Count workload above threshold
            movable = 0
            for _, deficit in renewable_deficit:
                if deficit > mid:
                    movable += deficit - mid

            if movable > target_movable:
                left = mid
            else:
                final_threshold = mid
                right = mid

        donor_hours = []
        recipient_hours = []

        for j in range(daily_df.shape[0]):
            deficit = max(
                0,
                daily_df["avg_dc_power_mw"].iloc[j] - daily_df["tot_renewable"].iloc[j],
            )
            if deficit > final_threshold:
                surplus = deficit - final_threshold
                donor_hours.append((j, surplus))
            else:
                capacity = min(
                    daily_df["tot_renewable"].iloc[j]
                    - daily_df["avg_dc_power_mw"].iloc[j],
                    max_capacity - daily_df["avg_dc_power_mw"].iloc[j],
                )
                if capacity > 0:
                    recipient_hours.append((j, capacity))

        # Sort donors by highest surplus (most urgent to move)
        donor_hours.sort(key=lambda x: x[1], reverse=True)
        # Sort recipients by highest capacity (most able to receive)
        recipient_hours.sort(key=lambda x: x[1], reverse=True)

        # Move workload from donors to recipients
        balanced_day_df = daily_df.copy()

        for donor_idx, surplus in donor_hours:
            movable = min(
                surplus,
                flexible_workload_ratio
                / 100
                * balanced_day_df["avg_dc_power_mw"].iloc[donor_idx],
            )

            if movable <= 0:
                continue

            for recipient_idx, capacity in recipient_hours:
                if capacity <= 0:
                    continue

                amount_to_move = min(movable, capacity)
                if amount_to_move <= 0:
                    continue

                # Move workload
                balanced_day_df["avg_dc_power_mw"].iloc[donor_idx] -= amount_to_move
                balanced_day_df["avg_dc_power_mw"].iloc[recipient_idx] += amount_to_move

                movable -= amount_to_move

                recipient_hours[recipient_hours.index((recipient_idx, capacity))] = (
                    recipient_idx,
                    capacity - amount_to_move,
                )

                if movable <= 0:
                    break

        balanced_df.append(balanced_day_df)

    final_balanced_df = pd.concat(balanced_df).sort_values(by=["index"])
    return final_balanced_df


def hybrid_cas_grid_mix(df_all, flexible_workload_ratio, max_capacity):
    """Hybrid search approach for carbon grid mix optimization using exponential followed by binary search"""
    balanced_df = []

    for i in range(0, df_all.shape[0], 24):
        daily_df = df_all[i : i + 24].copy()
        if daily_df.shape[0] < 24:
            break

        # Phase 1: Exponential search to find rough bounds
        # Sort hours by carbon intensity
        hourly_carbon = [
            (j, daily_df["carbon_intensity"].iloc[j]) for j in range(daily_df.shape[0])
        ]
        hourly_carbon.sort(key=lambda x: x[1], reverse=True)  # Highest carbon first

        # Calculate total movable workload
        target_movable = (
            sum(daily_df["avg_dc_power_mw"]) * flexible_workload_ratio / 100
        )

        # Find min and max carbon intensity for this day
        min_carbon = min([intensity for _, intensity in hourly_carbon])
        max_carbon = max([intensity for _, intensity in hourly_carbon])

        # Exponential search to find upper bound
        threshold = min_carbon + (max_carbon - min_carbon) * 0.1  # Start small
        prev_threshold = min_carbon

        while threshold < max_carbon:
            # Count workload above threshold
            movable = 0
            for _, intensity in hourly_carbon:
                if intensity > threshold:
                    hour_idx = next(
                        idx
                        for idx, (j, intens) in enumerate(hourly_carbon)
                        if intens == intensity
                    )
                    hourly_load = daily_df["avg_dc_power_mw"].iloc[
                        hourly_carbon[hour_idx][0]
                    ]
                    movable += hourly_load * flexible_workload_ratio / 100

            if movable < target_movable:
                break

            prev_threshold = threshold
            threshold = min_carbon + (threshold - min_carbon) * 2 

        # Phase 2: Binary search to refine threshold
        left = prev_threshold
        right = threshold
        final_threshold = 0

        for _ in range(
            7
        ):
            mid = (left + right) / 2

            # Count workload above threshold
            movable = 0
            for j, intensity in hourly_carbon:
                if intensity > mid:
                    hourly_load = daily_df["avg_dc_power_mw"].iloc[j]
                    movable += hourly_load * flexible_workload_ratio / 100

            if movable > target_movable:
                left = mid
            else:
                final_threshold = mid
                right = mid

        donor_hours = []
        recipient_hours = []

        for j in range(daily_df.shape[0]):
            if daily_df["carbon_intensity"].iloc[j] > final_threshold:
                # This is a donor hour (high carbon, needs to give away workload)
                donor_hours.append((j, daily_df["carbon_intensity"].iloc[j]))
            else:
                # This is a recipient hour (low carbon, can receive workload)
                capacity = max_capacity - daily_df["avg_dc_power_mw"].iloc[j]
                if capacity > 0:
                    recipient_hours.append((j, capacity))

        # Sort donors by highest carbon intensity (most urgent to move)
        donor_hours.sort(key=lambda x: x[1], reverse=True)
        # Sort recipients by lowest carbon intensity (best to receive)
        recipient_hours.sort(key=lambda x: daily_df["carbon_intensity"].iloc[x[0]])

        balanced_day_df = daily_df.copy()

        for donor_idx, _ in donor_hours:
            # Calculate how much this donor can actually give
            movable = (
                flexible_workload_ratio
                / 100
                * balanced_day_df["avg_dc_power_mw"].iloc[donor_idx]
            )

            if movable <= 0:
                continue

            for recipient_idx, capacity in recipient_hours:
                if capacity <= 0:
                    continue

                amount_to_move = min(movable, capacity)
                if amount_to_move <= 0:
                    continue

                # Move workload
                balanced_day_df["avg_dc_power_mw"].iloc[donor_idx] -= amount_to_move
                balanced_day_df["avg_dc_power_mw"].iloc[recipient_idx] += amount_to_move

                # Update remaining capacities
                movable -= amount_to_move

                recipient_hours[recipient_hours.index((recipient_idx, capacity))] = (
                    recipient_idx,
                    capacity - amount_to_move,
                )

                if movable <= 0:
                    break

        balanced_df.append(balanced_day_df)

    final_balanced_df = pd.concat(balanced_df).sort_values(by=["index"])
    return final_balanced_df
