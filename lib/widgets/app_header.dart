import 'package:flutter/material.dart';

import '../app.dart';

class AppHeader extends StatelessWidget {
  const AppHeader({
    super.key,
    required this.stageLabel,
    required this.progress,
    required this.onMenuPressed,
    required this.onInfoPressed,
  });

  final String stageLabel;
  final double progress;
  final VoidCallback onMenuPressed;
  final VoidCallback onInfoPressed;

  @override
  Widget build(BuildContext context) {
    return Material(
      color: Colors.white,
      child: Column(
        mainAxisSize: MainAxisSize.min,
        children: [
          SizedBox(
            width: double.infinity,
            height: 58,
            child: LayoutBuilder(
              builder: (context, constraints) {
                final brand = _HeaderBrand(onMenuPressed: onMenuPressed);
                final stage = _StageLabel(stageLabel: stageLabel);
                final infoButton = IconButton(
                  onPressed: onInfoPressed,
                  tooltip: 'Information about this stage',
                  icon: const Icon(Icons.info_outline_rounded),
                );

                if (constraints.maxWidth >= 600) {
                  return Stack(
                    alignment: Alignment.center,
                    children: [
                      Align(alignment: Alignment.centerLeft, child: brand),
                      Center(child: stage),
                      Align(
                        alignment: Alignment.centerRight,
                        child: infoButton,
                      ),
                    ],
                  );
                }

                return Row(
                  children: [
                    brand,
                    const Spacer(),
                    Flexible(child: stage),
                    const SizedBox(width: 8),
                    infoButton,
                  ],
                );
              },
            ),
          ),
          Semantics(
            label: 'Screening progress ${(progress * 100).round()} percent',
            child: LinearProgressIndicator(
              value: progress,
              minHeight: 4,
              color: AppColors.blue,
              backgroundColor: AppColors.softBlue,
            ),
          ),
        ],
      ),
    );
  }
}

class _HeaderBrand extends StatelessWidget {
  const _HeaderBrand({required this.onMenuPressed});

  final VoidCallback onMenuPressed;

  @override
  Widget build(BuildContext context) {
    return Row(
      mainAxisSize: MainAxisSize.min,
      children: [
        IconButton(
          onPressed: onMenuPressed,
          tooltip: 'Open menu',
          icon: const Icon(Icons.menu_rounded),
        ),
        const Icon(
          Icons.all_inclusive_rounded,
          color: AppColors.blue,
          size: 30,
          semanticLabel: 'Autism AI logo',
        ),
        const SizedBox(width: 8),
        const Text.rich(
          TextSpan(
            children: [
              TextSpan(
                text: 'Autism ',
                style: TextStyle(
                  color: AppColors.navy,
                  fontSize: 18,
                  fontWeight: FontWeight.w800,
                ),
              ),
              TextSpan(
                text: 'AI',
                style: TextStyle(
                  color: AppColors.blue,
                  fontSize: 18,
                  fontWeight: FontWeight.w800,
                ),
              ),
            ],
          ),
        ),
      ],
    );
  }
}

class _StageLabel extends StatelessWidget {
  const _StageLabel({required this.stageLabel});

  final String stageLabel;

  @override
  Widget build(BuildContext context) {
    return Text(
      stageLabel,
      maxLines: 1,
      overflow: TextOverflow.ellipsis,
      style: Theme.of(context).textTheme.labelLarge?.copyWith(
        color: AppColors.navy,
        fontWeight: FontWeight.w700,
      ),
    );
  }
}
