# oraska_learning_stress_test.ps1
Write-Host "Oraska v9.2.2 学习能力终极压测" -ForegroundColor Magenta
Write-Host "目标: 100 任务 -> Reward 0.9+" -ForegroundColor Cyan

$start_time = Get-Date
$success_count = 0
$rewards = @()

for ($i = 1; $i -le 100; $i++) {
    Write-Host "`n[$i/100] 任务执行中..." -ForegroundColor Yellow

    $body = @{
        description = "Design login API with JWT authentication - variation $i"
    } | ConvertTo-Json

    try {
        $response = Invoke-RestMethod -Uri "http://localhost:8000/tasks/execute" `
            -Method POST `
            -ContentType "application/json" `
            -Body $body `
            -TimeoutSec 120

        $success_count++
        $rewards += $response.reward

        Write-Host "  Task ID: $($response.task_id)" -ForegroundColor Cyan
        Write-Host "  Reward: $([math]::Round($response.reward, 3))" -ForegroundColor Green
        Write-Host "  Quality: $([math]::Round($response.quality, 3))" -ForegroundColor Green
        Write-Host "  Latency: $($response.latency_ms)ms" -ForegroundColor Gray

        if ($i % 20 -eq 0) {
            $avg = ($rewards | Measure-Object -Average).Average
            Write-Host "`n  平均 Reward (前 $i 次): $([math]::Round($avg, 3))" -ForegroundColor Magenta
        }

    } catch {
        Write-Host "  失败: $($_.Exception.Message)" -ForegroundColor Red
    }

    Start-Sleep -Seconds 2
}

$end_time = Get-Date
$duration = ($end_time - $start_time).ToString("hh\:mm\:ss")

Write-Host "`n学习能力测试完成！" -ForegroundColor Green
Write-Host "耗时: $duration" -ForegroundColor Cyan

$success_rate = [math]::Round($success_count / 100 * 100, 1)
Write-Host ("Success Rate: {0}/100 ({1}%)" -f $success_count, $success_rate) -ForegroundColor Cyan

try {
    $metrics = Invoke-RestMethod "http://localhost:8000/metrics"
    Write-Host "`n=== 系统学习指标 ===" -ForegroundColor Magenta
    Write-Host "总任务数: $($metrics.tasks)"
    Write-Host "平均奖励: $([math]::Round($metrics.avg_reward, 3))"
    Write-Host "成功率: $([math]::Round($metrics.success_rate * 100, 1))%"
    Write-Host "平均延迟: $([math]::Round($metrics.avg_latency_ms, 0))ms"

    Write-Host "`n=== Agent 状态 ===" -ForegroundColor Cyan
    for ($j = 0; $j -lt $metrics.agents.Count; $j++) {
        $a = $metrics.agents[$j]
        Write-Host ("Agent {0} | Loss: {1} | Value: {2} | Updates: {3}" -f 
            $j, 
            [math]::Round($a.policy_loss, 4), 
            [math]::Round($a.value, 3), 
            $a.updates)
    }
} catch {
    Write-Host "无法获取指标" -ForegroundColor Red
}

Write-Host "`n下一步：运行 'plot_curve.ps1' 生成学习曲线图" -ForegroundColor Yellow
