firstNCubeVolumes n = [m^3 | m <- [1..n]]
sumFirstNCubes = sum . firstNCubeVolumes

main = print(sumFirstNCubes 10)